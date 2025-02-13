#include "Isle.h"

Isle::Isle(std::string name) : name(name)
{
}

const std::string& Isle::getName() const
{
    return this->name;
}

Item Isle::getItem()
{
    return item;
}

void Isle::setItem(Item item)
{
    this->item = item;
}

bool Isle::operator==(const Isle &other) const
{
    // Compare by name
    return (this->name == other.name);
}

bool Isle::operator<(const Isle &other) const
{
    // Compare by name
    return (this->name < other.name);
}

bool Isle::operator>(const Isle &other) const
{
    // Compare by name
    return (this->name > other.name);
}

// Implementation of readFromFile
std::vector<Isle *> Isle::readFromFile(const std::string &filename)
{
    std::vector<Isle *> isles;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return isles;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip comments or empty lines
        if (line.empty() || line[0] == '#')
            continue;

        // Create a new Isle object and add it to the vector
        isles.push_back(new Isle(line));
    }

    file.close();
    return isles;
}

std::ostream &operator<<(std::ostream &os, const Isle &p)
{
    // Prints to terminal with color
    std::string einsteiniumColor = "\033[38;2;6;134;151m";
    std::string goldiumColor     = "\033[38;2;255;198;5m";
    std::string amazoniteColor   = "\033[38;2;169;254;255m";
    std::string resetColorTag    = "\033[0m";

    if (p.item == EINSTEINIUM)
        return (os << einsteiniumColor << p.name << resetColorTag);
    if (p.item == GOLDIUM)
        return (os << goldiumColor << p.name << resetColorTag);
    if (p.item == AMAZONITE)
        return (os << amazoniteColor << p.name << resetColorTag);
    return (os << p.name);
}
