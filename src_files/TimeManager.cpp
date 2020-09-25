//
// Created by finne on 7/11/2020.
//

#include "TimeManager.h"

#include "Board.h"

auto startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

/**
 * for exact timings
 */
TimeManager::TimeManager(int moveTime) {
    isSafeToStop = true;
    ignorePV     = true;
    forceStop    = false;

    timeToUse      = moveTime;
    upperTimeBound = moveTime;
    startTime      = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/**
 * for depth, infinite search etc.
 */
TimeManager::TimeManager() {
    isSafeToStop = true;
    ignorePV     = true;
    forceStop    = false;

    timeToUse      = 1 << 30;
    upperTimeBound = 1 << 30;

    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/**
 * for timecontrol
 * @param white
 * @param black
 * @param whiteInc
 * @param blackInc
 * @param movesToGo
 * @param board
 */
TimeManager::TimeManager(int white, int black, int whiteInc, int blackInc, int movesToGo, Board* board) {
    moveHistory  = new Move[256];
    scoreHistory = new Score[256];
    depthHistory = new Depth[256];
    historyCount = 0;
    isSafeToStop = true;
    ignorePV     = false;
    forceStop    = false;


    int division        = 30;

    timeToUse = board->getActivePlayer() == WHITE ? (int(white / division) + whiteInc) - 10
                                                  : (int(black / division) + blackInc) - 10;
    upperTimeBound = timeToUse;

    startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int TimeManager::elapsedTime() {
    auto                       end  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto                       diff = end - startTime;

    return diff;
    // std::cout << "measurement finished! [" << round(diff.count() * 1000) << " ms]" << std::endl;
}

TimeManager::~TimeManager() {

    if (ignorePV)
        return;

    delete[] moveHistory;
    delete[] scoreHistory;
    delete[] depthHistory;
}

void TimeManager::updatePV(Move move, Score score, Depth depth) {

    // dont keep track of pv changes if timing doesnt matter
    if (ignorePV)
        return;

    // store the move,score,depth in the arrays
    moveHistory[historyCount]  = move;
    scoreHistory[historyCount] = score;
    depthHistory[historyCount] = depth;

    // compute the safety to stop the search
    if (historyCount > 0){
        if (moveHistory[historyCount-1] == move) {
            timeToUse=timeToUse*0.93;
            if (timeToUse<upperTimeBound/2) timeToUse = upperTimeBound;
        } else {
            timeToUse = upperTimeBound;
        }
    }

    historyCount++;
}

void TimeManager::stopSearch() { forceStop = true; }

bool TimeManager::isTimeLeft() {

    //    return true;

    int elapsed = elapsedTime();

    //    std::cout << elapsed << " | " << forceStop << " | " << isSafeToStop << " | " << timeToUse << " | "<<
    //    upperTimeBound << " | " <<std::endl;

    // stop the search if requested and its safe
    if (forceStop && isSafeToStop)
        return false;

    // if we are above the maximum allowed time, stop
    if (elapsed >= upperTimeBound)
        return false;

    // if we have safe time left, continue searching
    if (elapsed < timeToUse)
        return true;

    // dont stop the search if its not safe
    if (!isSafeToStop)
        return true;

    return false;
}

void TimeManager::computeSafetyToStop() {
    // check if the pv didnt change but the score dropped heavily
    if (moveHistory[historyCount] == moveHistory[historyCount - 1]
        && scoreHistory[historyCount] < scoreHistory[historyCount - 1] - 60) {
        isSafeToStop = false;
        return;
    }

    // if the pv changed multiple times during this iteration, search deeper.
    Depth depth   = depthHistory[historyCount];
    int   index   = historyCount - 1;
    int   changes = 0;
    while (index >= 0) {
        if (depthHistory[index] != depth)
            break;

        changes++;

        index--;
    }

    if (changes >= 1) {
        isSafeToStop = false;
    } else {
        isSafeToStop = true;
    }
}
