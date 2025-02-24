# kd_tree.py
import numpy as np
from typing import List, Optional, Tuple

"""
Basic implementation of KD Tree (just for fun).

    TODO:
      1. Implement C++ module and extend python with pybind11.
      2. Add support for threading or processing for performance
         benefits.  
         
         Check scipy KDTree implementation.
"""

class KDNode:
    def __init__(self, point: np.ndarray, axis: int,
                 left: Optional['KDNode'] = None, right: Optional['KDNode'] = None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def build_kdtree(points: List[np.ndarray], depth: int = 0) -> Optional[KDNode]:
    """Just for fun"""
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2
    return KDNode(
        point=points[median],
        axis=axis,
        left=build_kdtree(points[:median], depth+1),
        right=build_kdtree(points[median+1:], depth+1))

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

def nearest_neighbor(node: KDNode, target: np.ndarray,
    best: Tuple[Optional[np.ndarray], float] = (None, float('inf'))) -> Tuple[Optional[np.ndarray], float]:
    if node is None:
        return best
    point = node.point
    axis = node.axis
    dist = euclidean_distance(target, point)
    if dist < best[1]:
        best = (point, dist)
    # Choose which side to search first:
    #   Case 1: If hyperplane is closer -> best distance -> search secondary
    #       else primary -> best distance
    diff = target[axis] - point[axis]
    primary = node.left if diff < 0 else node.right
    secondary = node.right if diff < 0 else node.left
    best = nearest_neighbor(primary, target, best)
    if abs(diff) < best[1]:
        best = nearest_neighbor(secondary, target, best)
    return best


if __name__ == '__main__':
    from tqdm.contrib.concurrent import process_map  # or thread_map

    def query_tree(tree: KDNode, target: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        return nearest_neighbor(tree, target)

    # Suppose targets is a list of target points
    targets = [np.random.rand(3) for _ in range(1000)]
    # Parallel query with a progress bar:
    results = process_map(lambda t: query_tree(tree, t),
                        targets,
                        chunksize=10,
                        disable=False,
                        desc="Parallel NN Queries")

    
    
    points = [np.random.rand(3) for _ in range(1000)]
    tree = build_kdtree(points)
    target = np.array([0.5, 0.5, 0.5])
    print(f'Nearest Neighbor: {nearest_neighbor(tree, target)}')
