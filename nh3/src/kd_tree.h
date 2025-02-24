// kd_tree.h
#pragma once
#include <vector>
#include <array>
#include <memory>
#include <limits>
#include <cmath>

using Point = std::array<double, 3>;

struct KDNode {
    Point point;
    int axis;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
};

double euclidean_distance(const Point &a, const Point &b) {
    double sum = 0;
    for (int i = 0; i < 3; ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

std::unique_ptr<KDNode> build_kdtree(std::vector<Point>::iterator begin,
                                     std::vector<Point>::iterator end,
                                     int depth = 0) {
    if (begin >= end) return nullptr;
    int axis = depth % 3;
    auto comparator = [axis](const Point &a, const Point &b) {
        return a[axis] < b[axis];
    };
    size_t len = std::distance(begin, end);
    auto mid = begin + len / 2;
    std::nth_element(begin, mid, end, comparator);
    auto node = std::make_unique<KDNode>();
    node->point = *mid;
    node->axis = axis;
    node->left = build_kdtree(begin, mid, depth + 1);
    node->right = build_kdtree(mid + 1, end, depth + 1);
    return node;
}

// Implement nearest neighbor similarly...
