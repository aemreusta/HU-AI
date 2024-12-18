#include "RealmShapers.h"
#include <cmath>
#include <algorithm>

ShaperTree::ShaperTree()
{
}

ShaperTree::~ShaperTree()
{
    // TODO: Free any dynamically allocated memory if necessary
    for (auto shaper : realmShapers)
    {
        delete shaper;
    }
    realmShapers.clear();
}

void ShaperTree::initializeTree(std::vector<RealmShaper *> shapers)
{
    // TODO: Insert innitial shapers to the tree
    for (auto shaper : shapers)
    {
        insert(shaper);
    }
}

int ShaperTree::getSize()
{
    // TODO: Return number of shapers in the tree
    return (this->realmShapers).size();
    // return 0;
}

std::vector<RealmShaper *> ShaperTree::getTree()
{
    return realmShapers;
}

bool ShaperTree::isValidIndex(int index)
{
    bool isValid = false;

    // TODO: Check if the index is valin in the tree
    if(index > 0 && index < (this->realmShapers).size())
        isValid = true;

    return isValid;
}

void ShaperTree::insert(RealmShaper *shaper)
{
    // TODO: Insert shaper to the tree
    this->realmShapers.push_back(shaper);
}

int ShaperTree::remove(RealmShaper *shaper)
{
    // TODO: Remove the player from tree if it exists
    // Make sure tree protects its form (complate binary tree) after deletion of a node
    // return index if found and removed
    // else

    int index = findIndex(shaper);
    if (index != -1){
        this->realmShapers.erase(this->realmShapers.begin() + index);
        return index;
    }
    
    return -1;
}

int ShaperTree::findIndex(RealmShaper *shaper)
{
    // return index in the tree if found
    // else
    for (int i = 0; i < this->realmShapers.size(); i++)
    {
        if (*this->realmShapers[i] == *shaper)
            return i;
    }
    return -1;
}

int ShaperTree::getDepth(RealmShaper *shaper)
{
    // return depth of the node in the tree if found
    // else

    int index = findIndex(shaper);
    if(index != -1){
        index++;
        int depth = 0;
        while(index > pow(2, depth) - 1){
            depth++;
        }
        return depth - 1;
    }

    return -1;
}

int ShaperTree::getDepth()
{
    int size = this->realmShapers.size();
    if(size > 0){
        int max_depth = 0;
        while(size > pow(2, max_depth) - 1){
            max_depth++;
        }
        return max_depth - 1;
    }

    // return total|max depth|height of the tree
    return 0;
}

RealmShaper ShaperTree::duel(RealmShaper *challenger, bool result)
{
    // TODO: Implement duel logic, return the victor
    // Use   std::cout << "[Duel] " << victorName << " won the duel" << std::endl;
    // Use   std::cout << "[Honour] " << "New honour points: ";
    // Use   std::cout << challengerName << "-" << challengerHonour << " ";
    // Use   std::cout << opponentName << "-" << opponentHonour << std::endl;
    // Use   std::cout << "[Duel] " << loserName << " lost all honour, delete" << std::endl;

    int challengerIndex = findIndex(challenger);
    if (challengerIndex == -1)
        return *challenger;

    RealmShaper *opponent = getParent(this->realmShapers[challengerIndex]);
    if (!opponent)
        return *challenger;

    if (result)
    {
        challenger->gainHonour();
        opponent->loseHonour();
        replace(challenger, opponent);
        std::cout << "[Duel] " << challenger->getName() << " won the duel" << std::endl;
    }
    else
    {
        challenger->loseHonour();
        std::cout << "[Duel] " << opponent->getName() << " won the duel" << std::endl;
    }

    std::cout << "[Honour] New honour points: " << challenger->getName() << "-" << challenger->getHonour() << " "
              << opponent->getName() << "-" << opponent->getHonour() << std::endl;

    if (challenger->getHonour() <= 0)
    {
        std::cout << "[Duel] " << challenger->getName() << " lost all honour, delete" << std::endl;
        remove(challenger);
    }

    return result ? *challenger : *opponent;
}

RealmShaper *ShaperTree::getParent(RealmShaper *shaper)
{
    RealmShaper *parent = nullptr;
    int index = findIndex(shaper);
    if(index != 0){
        index++;
        int parent_index = index / 2;
        return this->realmShapers[parent_index];
    }
    // TODO: return parent of the shaper

    return parent;
}

void ShaperTree::replace(RealmShaper *player_low, RealmShaper *player_high)
{
    // TODO: Change player_low and player_high's positions on the tree
    int index_low = findIndex(player_low);
    int index_high = findIndex(player_high);

    RealmShaper * temp = player_high;
    this->realmShapers[index_high] = player_low;
    this->realmShapers[index_low] = temp;
}

RealmShaper *ShaperTree::findPlayer(RealmShaper shaper)
{
    RealmShaper *foundShaper = nullptr;

    // TODO: Search shaper by object
    // Return the shaper if found
    // Return nullptr if shaper not found
    
    int index = findIndex(&shaper);
    if(index != 1)
        foundShaper = this->realmShapers[index];

    return foundShaper;
}

// Find shaper by name
RealmShaper *ShaperTree::findPlayer(std::string name)
{
    RealmShaper *foundShaper = nullptr;

    // TODO: Search shaper by name
    // Return the shaper if found
    // Return nullptr if shaper not found

    int size = this->realmShapers.size();
    for(int i = 0; i < size; i++){
        if(this->realmShapers[i]->getName() == name)
            foundShaper = this->realmShapers[i];
    }

    return foundShaper;
}

std::vector<std::string> ShaperTree::inOrderTraversal(int index)
{
    std::vector<std::string> result = {};
    // TODO: Implement inOrderTraversal in tree
    // Add all to a string vector
    // Return the vector

    // Define and implement as many helper functions as necessary for recursive implementation

    // Note: Since SheperTree is not an binary search tree,
    // in-order traversal will not give rankings in correct order
    // for correct order you need to implement level-order traversal
    // still you are to implement this function as well

    if (!isValidIndex(index))
        return result;

    auto left = inOrderTraversal(2 * index + 1);
    result.insert(result.end(), left.begin(), left.end());
    result.push_back(realmShapers[index]->getName());
    auto right = inOrderTraversal(2 * index + 2);
    result.insert(result.end(), right.begin(), right.end());

    return result;
}

std::vector<std::string> ShaperTree::preOrderTraversal(int index)
{
    std::vector<std::string> result = {};
    // TODO: Implement preOrderTraversal in tree
    // Add all to a string vector
    // Return the vector

    // Define and implement as many helper functions as necessary for recursive implementation

    if (!isValidIndex(index))
        return result;

    result.push_back(realmShapers[index]->getName());
    auto left = preOrderTraversal(2 * index + 1);
    result.insert(result.end(), left.begin(), left.end());
    auto right = preOrderTraversal(2 * index + 2);
    result.insert(result.end(), right.begin(), right.end());

    return result;
}

std::vector<std::string> ShaperTree::postOrderTraversal(int index)
{
    std::vector<std::string> result = {};
    // TODO: Implement postOrderTraversal in tree
    // Add all to a string vector
    // Return the vector

    // Define and implement as many helper functions as necessary for recursive implementation

    if (!isValidIndex(index))
        return result;

    auto left = postOrderTraversal(2 * index + 1);
    result.insert(result.end(), left.begin(), left.end());
    auto right = postOrderTraversal(2 * index + 2);
    result.insert(result.end(), right.begin(), right.end());
    result.push_back(realmShapers[index]->getName());

    return result;
}

void ShaperTree::preOrderTraversal(int index, std::ofstream &outFile)
{
    // TODO: Implement preOrderTraversal in tree
    // write nodes to output file

    // Define and implement as many helper functions as necessary for recursive implementation

    if (!isValidIndex(index))
        return;

    outFile << realmShapers[index]->getName() << " ";
    preOrderTraversal(2 * index + 1, outFile);
    preOrderTraversal(2 * index + 2, outFile);
}

void ShaperTree::breadthFirstTraversal(std::ofstream &outFile)
{
    // TODO: Implement level-order traversal
    // write nodes to output file

    // Define and implement as many helper functions as necessary

    for (auto shaper : realmShapers)
    {
        outFile << shaper->getName() << " ";
    }
}

void ShaperTree::displayTree()
{
    std::cout << "[Shaper Tree]" << std::endl;
    printTree(0, 0, "");
}

// Helper function to print tree with indentation
void ShaperTree::printTree(int index, int level, const std::string &prefix)
{
    if (!isValidIndex(index))
        return;

    std::cout << prefix << (level > 0 ? "   └---- " : "") << *realmShapers[index] << std::endl;
    int left = 0;  // TODO: Calculate left index
    int right = 0; // TODO: Calculate right index

    if (isValidIndex(left) || isValidIndex(right))
    {
        printTree(left, level + 1, prefix + (level > 0 ? "   │   " : "")); // ╎
        printTree(right, level + 1, prefix + (level > 0 ? "   │   " : ""));
    }
}

void ShaperTree::writeShapersToFile(const std::string &filename)
{
    // TODO: Write the shapers to filename output level by level
    // Use std::cout << "[Output] " << "Shapers have been written to " << filename << " according to rankings." << std::endl;

    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "[Error] Could not open file: " << filename << std::endl;
        return;
    }

    breadthFirstTraversal(outFile);
    outFile.close();
    std::cout << "[Output] Shapers have been written to " << filename << " according to rankings." << std::endl;

}

void ShaperTree::writeToFile(const std::string &filename)
{
    // TODO: Write the tree to filename output pre-order
    // Use std::cout << "[Output] " << "Tree have been written to " << filename << " in pre-order." << std::endl;

    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "[Error] Could not open file: " << filename << std::endl;
        return;
    }

    int currentLevel = 0;
    int nodesInCurrentLevel = 1;
    int nodesProcessed = 0;

    for (size_t i = 0; i < realmShapers.size(); ++i)
    {
        // Check if we've processed all nodes at the current level
        if (nodesProcessed == nodesInCurrentLevel)
        {
            outFile << std::endl; // Move to the next level
            currentLevel++;
            nodesInCurrentLevel = 1 << currentLevel; // Calculate the number of nodes in this level
            nodesProcessed = 0;
        }

        // Write the node or NULL if there is no node
        if (realmShapers[i] != nullptr)
        {
            outFile << realmShapers[i]->getName();
        }
        else
        {
            outFile << "NULL";
        }

        if (nodesProcessed < nodesInCurrentLevel - 1)
        {
            outFile << " "; // Add a space between nodes at the same level
        }

        nodesProcessed++;
    }

    // Handle trailing spaces or levels if required
    outFile.close();
    std::cout << "[Output] Shaper tree has been written to " << filename << " in the specified format." << std::endl;
}

