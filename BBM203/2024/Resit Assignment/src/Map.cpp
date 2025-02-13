#include "Map.h"

Map::Map()
{
    this->root = nullptr;
}

Map::~Map()
{
    // Free dynamically allocated memory in a breadth-first manner
    std::queue<MapNode *> nodes;
    if (root)
        nodes.push(root);

    while (!nodes.empty())
    {
        MapNode *current = nodes.front();
        nodes.pop();

        if (current->left)
            nodes.push(current->left);
        if (current->right)
            nodes.push(current->right);

        // Delete the Isle, then the MapNode
        delete current->isle;
        delete current;
    }
}

MapNode* Map::getRoot()
{
    return root;
}

bool Map::isRed(MapNode *node)
{
    return node != nullptr && node->isRed;
}

void Map::initializeMap(std::vector<Isle *> isles)
{
    // Insert initial Isles
    for (Isle *isle : isles)
    {
        insert(isle);
    }

    // Distribute items
    populateWithItems();
}

MapNode *Map::rotateRight(MapNode *current)
{
    if (!current || !current->left)
    {
        std::cerr << "[Right Rotation] Called on invalid node!" << std::endl;
        return current;
    }

    MapNode *newRoot = current->left;
    current->left = newRoot->right;
    newRoot->right = current;

    // Update colors to maintain LLRBT properties
    newRoot->isRed = current->isRed;
    current->isRed = true;

    return newRoot;
}

MapNode *Map::rotateLeft(MapNode *current)
{
    if (!current || !current->right)
    {
        std::cerr << "[Left Rotation] Called on invalid node!" << std::endl;
        return current;
    }

    MapNode *newRoot = current->right;
    current->right = newRoot->left;
    newRoot->left = current;

    // Update colors to maintain LLRBT properties
    newRoot->isRed = current->isRed;
    current->isRed = true;

    return newRoot;
}

MapNode *Map::insert(MapNode *node, Isle *isle)
{
    if (node == nullptr)
        return new MapNode(isle);

    if (*isle < *(node->isle))
        node->left = insert(node->left, isle);
    else if (*isle > *(node->isle))
        node->right = insert(node->right, isle);
    // If they are equal by name, do nothing (no duplicates).

    // Fix any right-leaning links
    if (isRed(node->right) && !isRed(node->left))
        node = rotateLeft(node);

    // Fix two red links in a row on the left
    if (isRed(node->left) && node->left && isRed(node->left->left))
        node = rotateRight(node);

    // Split 4-node
    if (isRed(node->left) && isRed(node->right))
    {
        node->isRed = true;
        if (node->left)
            node->left->isRed = false;
        if (node->right)
            node->right->isRed = false;
    }

    return node;
}

void Map::insert(Isle *isle)
{
    root = insert(root, isle);
    // Root is always black in a red-black tree
    if (root)
        root->isRed = false;
}

void Map::preOrderItemDrop(MapNode *current, int &count)
{
    if (!current)
        return;

    // Pre-order logic
    count++;
    if (count % 5 == 0)
    {
        current->isle->setItem(EINSTEINIUM);
        std::cout << "[Item Drop] EINSTEINIUM dropped on Isle: "
                  << current->isle->getName() << std::endl;
    }

    preOrderItemDrop(current->left, count);
    preOrderItemDrop(current->right, count);
}

void Map::postOrderItemDrop(MapNode *current, int &count)
{
    if (!current)
        return;

    // Post-order logic
    postOrderItemDrop(current->left, count);
    postOrderItemDrop(current->right, count);

    count++;
    if (count % 3 == 0)
    {
        current->isle->setItem(GOLDIUM);
        std::cout << "[Item Drop] GOLDIUM dropped on Isle: "
                  << current->isle->getName() << std::endl;
    }
}

MapNode *Map::findFirstEmptyIsle(MapNode *node)
{
    // BFS to find first Isle without an item
    if (!node)
        return nullptr;

    std::queue<MapNode *> q;
    q.push(node);

    while (!q.empty())
    {
        MapNode *current = q.front();
        q.pop();

        if (current->isle->getItem() == EMPTY)
            return current;

        if (current->left)
            q.push(current->left);
        if (current->right)
            q.push(current->right);
    }

    return nullptr;
}

void Map::dropItemBFS()
{
    if (!root)
    {
        std::cout << "[BFS Drop] No eligible Isle found for AMAZONITE drop." << std::endl;
        return;
    }

    // BFS to find first red node with an empty item
    std::queue<MapNode *> q;
    q.push(root);

    while (!q.empty())
    {
        MapNode *current = q.front();
        q.pop();

        if (current->isRed && current->isle->getItem() == EMPTY)
        {
            current->isle->setItem(AMAZONITE);
            std::cout << "[BFS Drop] AMAZONITE dropped on Isle: "
                      << current->isle->getName() << std::endl;
            return;
        }

        if (current->left)
            q.push(current->left);
        if (current->right)
            q.push(current->right);
    }

    std::cout << "[BFS Drop] No eligible Isle found for AMAZONITE drop." << std::endl;
}

void Map::displayMap()
{
    std::cout << "[World Map]" << std::endl;
    if (root)
        display(root, 0, 0);
}

int Map::getDepth(MapNode *node)
{
    if (!node)
        return -1;

    int depth = 0;
    MapNode *current = root;

    // Search from the root down to the node
    while (current)
    {
        if (*(current->isle) == *(node->isle))
            return depth;

        if (*(node->isle) < *(current->isle))
            current = current->left;
        else
            current = current->right;

        depth++;
    }
    return -1; // Node not found
}

void Map::populateWithItems()
{
    // Drop EINSTEINIUM by pre-order (every 5th)
    int preOrderCount = 0;
    preOrderItemDrop(root, preOrderCount);

    // Drop GOLDIUM by post-order (every 3rd)
    int postOrderCount = 0;
    postOrderItemDrop(root, postOrderCount);

    // Finally drop AMAZONITE by BFS
    dropItemBFS();
}

Isle *Map::findIsle(Isle isle)
{
    MapNode *node = findNode(isle);
    if (node)
        return node->isle;
    return nullptr;
}

Isle *Map::findIsle(std::string name)
{
    MapNode *node = findNode(name);
    if (node)
        return node->isle;
    return nullptr;
}

MapNode *Map::findNode(Isle isle)
{
    // Standard BST lookup
    MapNode *current = root;
    while (current)
    {
        if (*(current->isle) == isle)
            return current;
        else if (isle < *(current->isle))
            current = current->left;
        else
            current = current->right;
    }
    return nullptr;
}

MapNode *Map::findNode(std::string name)
{
    // Lookup by string
    MapNode *current = root;
    while (current)
    {
        if (current->isle->getName() == name)
            return current;
        else if (name < current->isle->getName())
            current = current->left;
        else
            current = current->right;
    }
    return nullptr;
}

void Map::display(MapNode *current, int depth, int state)
{
    if (current->left)
        display(current->left, depth + 1, 1);

    for (int i = 0; i < depth; i++)
        std::printf("     ");

    if (state == 1) // left
        std::printf("   ┌───");
    else if (state == 2) // right
        std::printf("   └───");

    std::cout << "[" << *current->isle << "] -  - ("
              << (current->isRed ? "\033[31mRed\033[0m" : "Black")
              << ")\n"
              << std::endl;

    if (current->right)
        display(current->right, depth + 1, 2);
}

void Map::writeToFile(const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Level-order traversal
    std::queue<MapNode *> q;
    if (root)
        q.push(root);

    while (!q.empty())
    {
        MapNode *current = q.front();
        q.pop();

        file << current->isle->getName() << "\t"
             << (current->isRed ? "Red" : "Black") << std::endl;

        if (current->left)
            q.push(current->left);
        if (current->right)
            q.push(current->right);
    }
    file.close();
}

void Map::writeIslesToFile(const std::string &filename)
{
    // Collect Isles by BFS
    std::vector<Isle *> isles;
    if (root)
    {
        std::queue<MapNode *> q;
        q.push(root);

        while (!q.empty())
        {
            MapNode *current = q.front();
            q.pop();
            isles.push_back(current->isle);

            if (current->left)
                q.push(current->left);
            if (current->right)
                q.push(current->right);
        }
    }

    // Sort Isles by name
    std::sort(isles.begin(), isles.end(), [](Isle *a, Isle *b) {
        return a->getName() < b->getName();
    });

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Write sorted Isles to file
    for (Isle *isle : isles)
    {
        file << isle->getName() << std::endl;
    }
    file.close();

    std::cout << "[Output] Isles have been written to " << filename
              << " in alphabetical order." << std::endl;
}
