#include "RealmShapers.h"
#include <cmath>
#include <algorithm>
#include <queue>

ShaperTree::ShaperTree() {}

ShaperTree::~ShaperTree()
{
    // Free the memory allocated for each RealmShaper
    for (auto shaper : realmShapers) {
        delete shaper;
    }
}

void ShaperTree::initializeTree(std::vector<RealmShaper *> shapers)
{
    for (auto shaper : shapers) {
        insert(shaper);
    }
}

int ShaperTree::getSize()
{
    return realmShapers.size();
}

std::vector<RealmShaper *> ShaperTree::getTree()
{
    return realmShapers;
}

bool ShaperTree::isValidIndex(int index)
{
    return (index >= 0 && index < (int)realmShapers.size());
}

void ShaperTree::insert(RealmShaper *shaper)
{
    realmShapers.push_back(shaper);
}

int ShaperTree::remove(RealmShaper *shaper)
{
    int index = findIndex(shaper);
    if (index != -1)
    {
        // Swap with the last element, then pop_back to keep the tree complete
        std::swap(realmShapers[index], realmShapers.back());
        realmShapers.pop_back();
        return index;
    }
    return -1;
}

int ShaperTree::findIndex(RealmShaper *shaper)
{
    auto it = std::find(realmShapers.begin(), realmShapers.end(), shaper);
    if (it != realmShapers.end())
    {
        return std::distance(realmShapers.begin(), it);
    }
    return -1;
}

int ShaperTree::getDepth(RealmShaper *shaper)
{
    int index = findIndex(shaper);
    if (index == -1)
        return -1;
    return std::floor(std::log2(index + 1));
}

int ShaperTree::getDepth()
{
    if (realmShapers.empty())
        return 0;
    return std::floor(std::log2(realmShapers.size())) + 1;
}

RealmShaper *ShaperTree::getParent(RealmShaper *shaper)
{
    int index = findIndex(shaper);
    if (index <= 0)
        return nullptr;
    return realmShapers[(index - 1) / 2];
}

RealmShaper *ShaperTree::findPlayer(RealmShaper shaper)
{
    for (auto &found : realmShapers)
    {
        if (*found == shaper)
            return found;
    }
    return nullptr;
}

RealmShaper *ShaperTree::findPlayer(std::string name)
{
    for (auto &found : realmShapers)
    {
        if (found->getName() == name)
            return found;
    }
    return nullptr;
}

std::vector<std::string> ShaperTree::inOrderTraversal(int index)
{
    std::vector<std::string> result;
    if (isValidIndex(index))
    {
        int leftIndex = 2 * index + 1;
        int rightIndex = 2 * index + 2;

        if (isValidIndex(leftIndex))
        {
            auto leftResult = inOrderTraversal(leftIndex);
            result.insert(result.end(), leftResult.begin(), leftResult.end());
        }

        result.push_back(realmShapers[index]->getName());

        if (isValidIndex(rightIndex))
        {
            auto rightResult = inOrderTraversal(rightIndex);
            result.insert(result.end(), rightResult.begin(), rightResult.end());
        }
    }
    return result;
}

void ShaperTree::preOrderHelper(int index, std::vector<std::string> &result)
{
    if (!isValidIndex(index))
        return;

    result.push_back(realmShapers[index]->getName());

    int leftIndex = 2 * index + 1;
    if (leftIndex < (int)realmShapers.size())
        preOrderHelper(leftIndex, result);

    int rightIndex = 2 * index + 2;
    if (rightIndex < (int)realmShapers.size())
        preOrderHelper(rightIndex, result);
}

std::vector<std::string> ShaperTree::preOrderTraversal(int index)
{
    std::vector<std::string> result;
    preOrderHelper(index, result);
    return result;
}

void ShaperTree::postOrderHelper(int index, std::vector<std::string> &result)
{
    if (!isValidIndex(index))
        return;

    int leftIndex = 2 * index + 1;
    if (leftIndex < (int)realmShapers.size())
        postOrderHelper(leftIndex, result);

    int rightIndex = 2 * index + 2;
    if (rightIndex < (int)realmShapers.size())
        postOrderHelper(rightIndex, result);

    result.push_back(realmShapers[index]->getName());
}

std::vector<std::string> ShaperTree::postOrderTraversal(int index)
{
    std::vector<std::string> result;
    postOrderHelper(index, result);
    return result;
}

void ShaperTree::breadthFirstTraversal(std::ofstream &outFile)
{
    if (realmShapers.empty())
        return;

    std::queue<int> q;
    q.push(0);

    while (!q.empty())
    {
        int currentIndex = q.front();
        q.pop();

        if (!isValidIndex(currentIndex))
            continue;

        outFile << realmShapers[currentIndex]->getName() << std::endl;

        int leftIndex = 2 * currentIndex + 1;
        int rightIndex = 2 * currentIndex + 2;

        if (leftIndex < (int)realmShapers.size())
            q.push(leftIndex);
        if (rightIndex < (int)realmShapers.size())
            q.push(rightIndex);
    }
}

void ShaperTree::displayTree()
{
    std::cout << "[Shaper Tree]" << std::endl;
    printTree(0, 0, "");
}

int ShaperTree::getLeftChildIndex(int index)
{
    int left = 2 * index + 1;
    return isValidIndex(left) ? left : -1;
}

int ShaperTree::getRightChildIndex(int index)
{
    int right = 2 * index + 2;
    return isValidIndex(right) ? right : -1;
}

void ShaperTree::printTree(int index, int level, const std::string &prefix)
{
    if (!isValidIndex(index))
        return;

    std::cout << prefix << (level > 0 ? "   └---- " : "") << *realmShapers[index] << std::endl;

    int left = getLeftChildIndex(index);
    int right = getRightChildIndex(index);

    if (left != -1 || right != -1)
    {
        printTree(left, level + 1, prefix + (level > 0 ? "   │   " : ""));
        printTree(right, level + 1, prefix + (level > 0 ? "   │   " : ""));
    }
}
