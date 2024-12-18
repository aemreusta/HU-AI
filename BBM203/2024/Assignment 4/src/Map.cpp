#include "Map.h"
#include <functional>

Map::Map()
{
    this->root = nullptr;
}
Map::~Map()
{
    // TODO: Free any dynamically allocated memory if necessary
    std::function<void(MapNode *)> deleteTree = [&](MapNode *node) {
        if (!node) return;
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    };

    deleteTree(root);
    root = nullptr;
}

void Map::initializeMap(std::vector<Isle *> isles)
{
    // TODO: Insert innitial isles to the tree
    // Then populate with Goldium and Einstainium items
    for (Isle *isle : isles)
        insert(isle);

    populateWithItems();
}

MapNode *Map::rotateRight(MapNode *current)
{
    // TODO: Perform right rotation according to AVL
    // return necessary new root
    // Use std::cerr << "[Right Rotation] " << "Called on invalid node!" << std::endl;

    if (!current || !current->left) {
        std::cerr << "[Right Rotation] " << "Called on invalid node!" << std::endl;
        return current;
    }

    MapNode *newRoot = current->left;
    current->left = newRoot->right;
    newRoot->right = current;

    // Update heights
    current->height = std::max(height(current->left), height(current->right)) + 1;
    newRoot->height = std::max(height(newRoot->left), height(newRoot->right)) + 1;

    return newRoot;
}

MapNode *Map::rotateLeft(MapNode *current)
{
    // TODO: Perform left rotation according to AVL
    // return necessary new root
    // Use std::cerr << "[Left Rotation] " << "Called on invalid node!" << std::endl;

    if (!current || !current->right) {
        std::cerr << "[Left Rotation] " << "Called on invalid node!" << std::endl;
        return current;
    }

    MapNode *newRoot = current->right;
    current->right = newRoot->left;
    newRoot->left = current;

    // Update heights
    current->height = std::max(height(current->left), height(current->right)) + 1;
    newRoot->height = std::max(height(newRoot->left), height(newRoot->right)) + 1;

    return newRoot;
}

int Map::calculateMinMapDepthAccess(int playerDepth, int totalShaperTreeHeight, int totalMapDepth)
{
    return (int)totalMapDepth * ((double)playerDepth / totalShaperTreeHeight);
}

int Map::height(MapNode *node)
{
    // TODO: Return height of the node
    if(node)
        return node->height;
    return 0;
}

MapNode *Map::insert(MapNode *node, Isle *isle)
{
    MapNode *newNode = nullptr;

    // TODO: Recursively insert isle to the tree
    // returns inserted node

    if (!node) 
        return new MapNode(isle);

    if (isle->getName() < node->isle->getName())
        node->left = insert(node->left, isle);
    else if (isle->getName() > node->isle->getName())
        node->right = insert(node->right, isle);
    else
        return node; // Duplicates are not allowed

    // Update height
    node->height = std::max(height(node->left), height(node->right)) + 1;

    // Check balance factor
    int balance = height(node->left) - height(node->right);

    // Perform rotations if unbalanced
    if (balance > 1 && isle->getName() < node->left->isle->getName())
        return rotateRight(node);

    if (balance < -1 && isle->getName() > node->right->isle->getName())
        return rotateLeft(node);

    if (balance > 1 && isle->getName() > node->left->isle->getName()) {
        node->left = rotateLeft(node->left);
        return rotateRight(node);
    }

    if (balance < -1 && isle->getName() < node->right->isle->getName()) {
        node->right = rotateRight(node->right);
        return rotateLeft(node);
    }

    return node;

    return newNode;
}

void Map::insert(Isle *isle)
{
    root = insert((root), isle);
    // you might need to insert some checks / functions here depending on your implementation
}

MapNode *Map::remove(MapNode *node, Isle *isle)
{
    // TODO: Recursively delete isle from the tree
    // Will be called if there is overcrowding
    // returns node
    // Use std::cout << "[Remove] " << "Tree is Empty" << std::endl;

    if (!node) {
        std::cout << "[Remove] Tree is Empty" << std::endl;
        return nullptr;
    }

    // Locate the node to remove
    if (isle->getName() < node->isle->getName())
        node->left = remove(node->left, isle);
    else if (isle->getName() > node->isle->getName())
        node->right = remove(node->right, isle);
    else {
        // Node found
        if (!node->left || !node->right) {
            // One or no child
            MapNode *temp = node->left ? node->left : node->right;
            delete node;
            return temp;
        } else {
            // Two children: Find inorder successor
            MapNode *successor = node->right;
            while (successor->left)
                successor = successor->left;

            node->isle = successor->isle;
            node->right = remove(node->right, successor->isle);
        }
    }

    // Update height and rebalance
    node->height = 1 + std::max(height(node->left), height(node->right));

    int balance = height(node->left) - height(node->right);

    // Left-heavy cases
    if (balance > 1 && height(node->left->left) >= height(node->left->right))
        return rotateRight(node);
    if (balance > 1 && height(node->left->left) < height(node->left->right)) {
        node->left = rotateLeft(node->left);
        return rotateRight(node);
    }

    // Right-heavy cases
    if (balance < -1 && height(node->right->right) >= height(node->right->left))
        return rotateLeft(node);
    if (balance < -1 && height(node->right->right) < height(node->right->left)) {
        node->right = rotateRight(node->right);
        return rotateLeft(node);
    }

    return node;
}

void Map::remove(Isle *isle)
{
    root = remove((root), isle);
    // you might need to insert some checks / functions here depending on your implementation
}

void Map::preOrderItemDrop(MapNode *current, int &count)
{
    // TODO: Drop EINSTEINIUM according to rules
    // Use std::cout << "[Item Drop] " << "EINSTEINIUM dropped on Isle: " << current->isle->getName() << std::endl;

    if (!current) return;

    count++;
    if (count % 5 == 0) {
        current->isle->setItem(Item::EINSTEINIUM);
        std::cout << "[Item Drop] " << "EINSTEINIUM dropped on Isle: " << current->isle->getName() << std::endl;
    }

    preOrderItemDrop(current->left, count);
    preOrderItemDrop(current->right, count);
}

// to Display the values by Post Order Method .. left - right - node
void Map::postOrderItemDrop(MapNode *current, int &count)
{
    // TODO: Drop GOLDIUM according to rules
    // Use  std::cout << "[Item Drop] " << "GOLDIUM dropped on Isle: " << current->isle->getName() << std::endl;

    if (!current) return;

    postOrderItemDrop(current->left, count);
    postOrderItemDrop(current->right, count);

    count++;
    if (count % 3 == 0) {
        current->isle->setItem(Item::GOLDIUM);
        std::cout << "[Item Drop] " << "GOLDIUM dropped on Isle: " << current->isle->getName() << std::endl;
    }
}

// MapNode *Map::findFirstEmptyIsle(MapNode *node)
// {
//     // TODO: Find first Isle with no item
// }

void Map::dropItemBFS()
{
    // TODO: Drop AMAZONITE according to rules
    // Use std::cout << "[BFS Drop] " << "AMAZONITE dropped on Isle: " << targetNode->isle->getName() << std::endl;
    // Use std::cout << "[BFS Drop] " << "No eligible Isle found for AMAZONITE drop." << std::endl;
    if (!root) {
        std::cout << "[BFS Drop] " << "No eligible Isle found for AMAZONITE drop." << std::endl;
        return;
    }

    std::queue<MapNode *> q;
    q.push(root);

    while (!q.empty()) {
        MapNode *current = q.front();
        q.pop();

        if (current->isle->getItem() == Item::EMPTY) {
            current->isle->setItem(Item::AMAZONITE);
            std::cout << "[BFS Drop] " << "AMAZONITE dropped on Isle: " << current->isle->getName() << std::endl;
            return;
        }

        if (current->left) q.push(current->left);
        if (current->right) q.push(current->right);
    }

    std::cout << "[BFS Drop] " << "No eligible Isle found for AMAZONITE drop." << std::endl;
}

void Map::displayMap()
{
    std::cout << "[World Map]" << std::endl;
    display(root, 0, 0);
}

int Map::getDepth(MapNode *node)
{
    // TODO: Return node depth if found, else
    if (!root || !node) 
        return -1;

    MapNode *current = root;
    int depth = 0;

    while (current) {
        if (node->isle->getName() == current->isle->getName())
            return depth;

        if (node->isle->getName() < current->isle->getName())
            current = current->left;
        else
            current = current->right;

        depth++;
    }

    return -1;
}

// Function to calculate the depth of a specific node in the AVL tree
int Map::getIsleDepth(Isle *isle)
{
    // TODO: Return node depth by isle if found, else
    MapNode *node = findNode(*isle);
    if(node)
        return getDepth(node);
    return -1;
}

int Map::getDepth()
{
    // TODO: Return max|total depth of tree
    if (root != nullptr){
        // compute the height of left and right subtrees
        int lHeight = height(root->left);
        int rHeight = height(root->right);

        if(lHeight >= rHeight)
            return lHeight + 1;
        else
            return rHeight + 1;    
    }
    return -1;
}

void Map::populateWithItems()
{
    // TODO: Distribute fist GOLDIUM than EINSTEINIUM
    if (!root) {
        std::cout << "[Populate] " << "Tree is empty. Cannot populate items." << std::endl;
        return;
    }

    int preOrderCount = 0;
    preOrderItemDrop(root, preOrderCount); // Populate with EINSTEINIUM

    int postOrderCount = 0;
    postOrderItemDrop(root, postOrderCount); // Populate with GOLDIUM

    std::cout << "[Populate] " << "Items have been distributed." << std::endl;
}

Isle *Map::findIsle(Isle isle)
{
    // TODO: Find isle by value
    MapNode *node = findNode(isle);
    if(node) 
        return node->isle;
    return nullptr;
}

Isle *Map::findIsle(std::string name)
{
    // TODO: Find isle by name
    MapNode *node = findNode(name);
    if(node) 
        return node->isle ;
    return nullptr;
}

MapNode *Map::findNode(Isle isle)
{
    // TODO: Find node by value
    MapNode *current = root;

    while (current) {
        if (current->isle->getName() == isle.getName())
            return current;

        if (isle.getName() < current->isle->getName())
            current = current->left;
        else
            current = current->right;
    }

    return nullptr;
}

MapNode *Map::findNode(std::string name)
{
    // TODO: Find node by name
    MapNode *current = root;

    while (current) {
        if (current->isle->getName() == name)
            return current;

        if (name < current->isle->getName())
            current = current->left;
        else
            current = current->right;
    }

    return nullptr;
}

void Map::display(MapNode *current, int depth, int state)
{
    // SOURCE:

    if (current->left)
        display(current->left, depth + 1, 1);

    for (int i = 0; i < depth; i++)
        printf("     ");

    if (state == 1) // left
        printf("   ┌───");
    else if (state == 2) // right
        printf("   └───");

    std::cout << "[" << *current->isle << "] - (" << current->height << ")\n"
              << std::endl;

    if (current->right)
        display(current->right, depth + 1, 2);
}

void Map::writeToFile(const std::string &filename)
{
    // TODO: Write the tree to filename output level by level
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[WriteToFile] Could not open file: " << filename << std::endl;
        return;
    }

    std::queue<MapNode *> q;
    q.push(root);

    file << "[World Map - Level Order Traversal]\n";
    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; i++) {
            MapNode *current = q.front();
            q.pop();

            if (current) {
                file << current->isle->getName() << " ";
                q.push(current->left);
                q.push(current->right);
            } else {
                file << "null ";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "[WriteToFile] Tree written to " << filename << std::endl;
}

void Map::writeIslesToFile(const std::string &filename)
{
    // TODO: Write Isles to output file in alphabetical order
    // Use std::cout << "[Output] " << "Isles have been written to " << filename << " in in alphabetical order." << std::endl;
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[WriteIslesToFile] Could not open file: " << filename << std::endl;
        return;
    }

    std::function<void(MapNode *)> inOrderTraversal = [&](MapNode *node) {
        if (!node) return;
        inOrderTraversal(node->left);
        file << node->isle->getName() << "\n";
        inOrderTraversal(node->right);
    };

    file << "[Isles - Alphabetical Order]\n";
    inOrderTraversal(root);
    file.close();
    std::cout << "[Output] Isles have been written to " << filename << " in alphabetical order." << std::endl;
}