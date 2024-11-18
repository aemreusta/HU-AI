#include "LeaderboardEntry.h"

// Constructor to initialize a leaderboard entry
LeaderboardEntry::LeaderboardEntry(unsigned long score,
                                   time_t lastPlayed,
                                   const string &playerName) :
        score(score),
        last_played(lastPlayed),
        player_name(playerName) {
    // No further initializations required
}
