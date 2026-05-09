//! Draft-tree primitives shared by Medusa, EAGLE, and any other
//! tree-style speculative decoder.
//!
//! ## Why a tree?
//!
//! Vanilla SD considers exactly one draft trajectory per verification round —
//! `k` tokens in a line. Medusa / EAGLE generalise this: the draft proposes a
//! *tree* of candidates (root = the last committed token, branches = different
//! continuations) and the target verifies the entire tree in one forward pass.
//! Each tree node reuses its ancestors' KV state, so the cost is `O(tree_size)`
//! instead of `O(tree_size^2)`.
//!
//! ## What this module provides
//!
//! - [`DraftTree`]: a compact parent-pointer representation
//! - [`DraftTree::attention_mask_bool`]: the per-node attention mask (`true`
//!   iff node `i` may attend to position `j`)
//! - [`DraftTree::position_ids`]: the per-node positional offset to feed RoPE
//! - [`DraftTree::path_to`]: the chain of node indices from root to a leaf,
//!   useful for replay after the target picks a winning path
//!
//! These are pure data-structure operations — no candle dependency, no
//! tensors. The tensor-side glue (turning a bool mask into a `[1, 1, n, n]`
//! attention bias) lives in the model-specific decoder once Phase 2 wires
//! tree attention into Qwen2 / Llama / etc.

#![allow(clippy::needless_range_loop)]

use crate::{Error, Result};

/// A draft tree rooted at a single (already-committed) token.
///
/// Nodes are stored in BFS order: the root is index 0, then all depth-1 nodes,
/// then depth-2, and so on. Every non-root node has a strictly-smaller parent
/// index, which makes ancestor traversal a simple `while`-loop.
#[derive(Debug, Clone)]
pub struct DraftTree {
    /// Token id at each node. `tokens[0]` is the root token.
    tokens: Vec<u32>,
    /// `parents[i]` is the parent node index of node `i`. The root has
    /// `parents[0] == 0` (self-loop sentinel — easier to handle than `None`).
    parents: Vec<usize>,
}

impl DraftTree {
    /// Construct from a list of `(parent_index, token)` pairs.
    ///
    /// The first entry is treated as the root (its `parent_index` field is
    /// ignored). All subsequent entries must reference a parent that already
    /// exists (i.e. has a strictly-smaller index in the slice).
    pub fn from_parent_table(nodes: &[(usize, u32)]) -> Result<Self> {
        if nodes.is_empty() {
            return Err(Error::Sampling(
                "DraftTree must have at least a root".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(nodes.len());
        let mut parents = Vec::with_capacity(nodes.len());
        // Root.
        tokens.push(nodes[0].1);
        parents.push(0);
        // Non-root nodes.
        for (i, &(p, tok)) in nodes.iter().enumerate().skip(1) {
            if p >= i {
                return Err(Error::Sampling(format!(
                    "node {i} has parent index {p}, which is not strictly smaller",
                )));
            }
            tokens.push(tok);
            parents.push(p);
        }
        Ok(Self { tokens, parents })
    }

    /// Construct a *linear* (vanilla-SD) tree: root → tok_1 → tok_2 → ... → tok_k.
    pub fn linear(root: u32, tail: &[u32]) -> Self {
        let mut tokens = Vec::with_capacity(tail.len() + 1);
        let mut parents = Vec::with_capacity(tail.len() + 1);
        tokens.push(root);
        parents.push(0);
        for (i, &t) in tail.iter().enumerate() {
            tokens.push(t);
            parents.push(i); // previous node
        }
        Self { tokens, parents }
    }

    /// Number of nodes (root + draft positions).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the tree has only the root.
    pub fn is_empty(&self) -> bool {
        self.tokens.len() <= 1
    }

    /// Token at node `i`.
    pub fn token_at(&self, i: usize) -> u32 {
        self.tokens[i]
    }

    /// All tokens in BFS order (suitable as `input_ids` for a model forward).
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Parent index of node `i`. The root returns `0` (self).
    pub fn parent_of(&self, i: usize) -> usize {
        self.parents[i]
    }

    /// Iterator over all ancestors of `i` (including `i` itself), root last.
    pub fn ancestors(&self, mut i: usize) -> Vec<usize> {
        let mut out = vec![i];
        while i != 0 {
            i = self.parents[i];
            out.push(i);
        }
        out
    }

    /// Depth of node `i` (root = 0).
    pub fn depth_of(&self, i: usize) -> usize {
        let mut d = 0;
        let mut cur = i;
        while cur != 0 {
            cur = self.parents[cur];
            d += 1;
        }
        d
    }

    /// Per-node depth, indexed by node id. Useful for RoPE position_ids when
    /// the prefix length (number of committed tokens) is added on top.
    pub fn position_ids(&self, prefix_len: usize) -> Vec<usize> {
        (0..self.len())
            .map(|i| prefix_len + self.depth_of(i))
            .collect()
    }

    /// Build the per-node attention mask for the *tree positions only* (the
    /// shared prefix is handled separately by the caller — every tree node
    /// attends to the entire prefix unconditionally).
    ///
    /// `mask[i][j] == true` iff `j` is an ancestor of `i` (inclusive). This is
    /// the structural part of the tree-attention bias.
    pub fn attention_mask_bool(&self) -> Vec<Vec<bool>> {
        let n = self.len();
        let mut mask = vec![vec![false; n]; n];
        for i in 0..n {
            for j in self.ancestors(i) {
                mask[i][j] = true;
            }
        }
        mask
    }

    /// All root-to-leaf paths as Vec of node-index chains. Leaves are nodes
    /// that no other node points to as parent.
    pub fn paths(&self) -> Vec<Vec<usize>> {
        let mut is_leaf = vec![true; self.len()];
        for &p in self.parents.iter().skip(1) {
            is_leaf[p] = false;
        }
        let mut out = Vec::new();
        for (i, &leaf) in is_leaf.iter().enumerate() {
            if leaf {
                let mut chain = self.ancestors(i);
                chain.reverse(); // now root..leaf
                out.push(chain);
            }
        }
        out
    }

    /// Path of node indices from the root to `target`, in order.
    pub fn path_to(&self, target: usize) -> Vec<usize> {
        let mut chain = self.ancestors(target);
        chain.reverse();
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_tree_is_a_chain() {
        let t = DraftTree::linear(10, &[20, 30, 40]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.tokens(), &[10, 20, 30, 40]);
        assert_eq!(t.parent_of(0), 0);
        assert_eq!(t.parent_of(1), 0);
        assert_eq!(t.parent_of(2), 1);
        assert_eq!(t.parent_of(3), 2);
        assert_eq!(t.depth_of(3), 3);
        assert_eq!(t.paths(), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn branching_tree_paths() {
        // Tree:
        //         0
        //       /   \
        //      1     2
        //     / \    |
        //    3   4   5
        let t = DraftTree::from_parent_table(&[
            (0, 100), // root
            (0, 11),
            (0, 12),
            (1, 23),
            (1, 24),
            (2, 35),
        ])
        .unwrap();
        assert_eq!(t.len(), 6);
        assert_eq!(t.depth_of(3), 2);
        assert_eq!(t.depth_of(5), 2);
        let mut paths = t.paths();
        paths.sort_by_key(|p| (p.len(), p.clone()));
        assert_eq!(paths, vec![vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 5]]);
    }

    #[test]
    fn linear_mask_is_lower_triangular() {
        let t = DraftTree::linear(10, &[20, 30, 40]);
        let m = t.attention_mask_bool();
        // Linear tree → mask should be lower-triangular causal.
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i][j], j <= i, "expected causal at ({i},{j})");
            }
        }
    }

    #[test]
    fn branching_mask_blocks_siblings() {
        // Same tree as branching_tree_paths.
        let t =
            DraftTree::from_parent_table(&[(0, 100), (0, 11), (0, 12), (1, 23), (1, 24), (2, 35)])
                .unwrap();
        let m = t.attention_mask_bool();

        // Node 3 (left grandchild of 1) should attend to 0, 1, 3 only.
        assert!(m[3][0] && m[3][1] && m[3][3]);
        assert!(!m[3][2], "node 3 must NOT see sibling-of-parent (2)");
        assert!(!m[3][4], "node 3 must NOT see sibling (4)");
        assert!(!m[3][5], "node 3 must NOT see other-branch leaf (5)");

        // Symmetric checks for node 5.
        assert!(m[5][0] && m[5][2] && m[5][5]);
        assert!(!m[5][1] && !m[5][3] && !m[5][4]);
    }

    #[test]
    fn position_ids_offset_by_prefix() {
        let t = DraftTree::linear(0, &[1, 2, 3]);
        let pos = t.position_ids(7);
        assert_eq!(pos, vec![7, 8, 9, 10]);
    }

    #[test]
    fn rejects_forward_parent_reference() {
        let bad = [(0, 0u32), (5, 1)]; // node 1 references parent 5 which doesn't exist yet
        assert!(DraftTree::from_parent_table(&bad).is_err());
    }

    #[test]
    fn rejects_empty_tree() {
        assert!(DraftTree::from_parent_table(&[]).is_err());
    }

    #[test]
    fn path_to_walks_root_first() {
        let t = DraftTree::from_parent_table(&[(0, 0), (0, 1), (1, 2), (2, 3)]).unwrap();
        assert_eq!(t.path_to(3), vec![0, 1, 2, 3]);
    }
}
