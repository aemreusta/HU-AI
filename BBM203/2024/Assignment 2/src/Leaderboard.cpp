#include "Leaderboard.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

// Read the stored leaderboard status from the given file
void Leaderboard::read_from_file(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        return; // If the file doesn't exist, start with an empty leaderboard
    }

    unsigned long score;
    time_t lastPlayed;
    string playerName;

    while (file >> score >> lastPlayed >> ws && getline(file, playerName)) {
        auto *new_entry = new LeaderboardEntry(score, lastPlayed, playerName);
        insert(new_entry); // Insert into the leaderboard in the correct order
    }

    file.close();
}

// Write the latest leaderboard status to the given file
void Leaderboard::write_to_file(const string &filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to write to file " << filename << endl;
        return;
    }

    LeaderboardEntry *current = head_leaderboard_entry;
    while (current != nullptr) {
        file << current->score << " " << current->last_played << " " << current->player_name << "\n";
        current = current->next;
    }

    file.close();
}

// Print the current leaderboard status to the standard output
void Leaderboard::print_leaderboard() {
    cout << "Leaderboard\n-----------\n";

    LeaderboardEntry *current = head_leaderboard_entry;
    int rank = 1;

    while (current != nullptr) {
        // Convert timestamp to human-readable format
        tm *time_info = localtime(&current->last_played);
        stringstream timestamp;
        timestamp << put_time(time_info, "%H:%M:%S/%d.%m.%Y");

        cout << rank << ". " << current->player_name << " " << current->score << " " << timestamp.str() << "\n";
        rank++;
        current = current->next;
    }
}

// Insert a new LeaderboardEntry instance into the leaderboard
void Leaderboard::insert(LeaderboardEntry *new_entry) {
    if (!new_entry) return;

    if (!head_leaderboard_entry || new_entry->score > head_leaderboard_entry->score) {
        // Insert at the head if the list is empty or the new entry has the highest score
        new_entry->next = head_leaderboard_entry;
        head_leaderboard_entry = new_entry;
    } else {
        LeaderboardEntry *current = head_leaderboard_entry;

        // Traverse to find the correct position
        while (current->next && current->next->score >= new_entry->score) {
            current = current->next;
        }

        // Insert the new entry in the correct position
        new_entry->next = current->next;
        current->next = new_entry;
    }

    // Trim the list to MAX_LEADERBOARD_SIZE (10 entries)
    LeaderboardEntry *current = head_leaderboard_entry;
    int count = 1;
    while (current && current->next) {
        if (count == MAX_LEADERBOARD_SIZE) {
            // Delete excess entries
            LeaderboardEntry *to_delete = current->next;
            current->next = nullptr;
            delete to_delete;
            break;
        }
        count++;
        current = current->next;
    }
}

// Free dynamically allocated memory used for storing leaderboard entries
Leaderboard::~Leaderboard() {
    while (head_leaderboard_entry) {
        LeaderboardEntry *to_delete = head_leaderboard_entry;
        head_leaderboard_entry = head_leaderboard_entry->next;
        delete to_delete;
    }
}
