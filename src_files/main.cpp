
#include "Bitboard.h"
#include "Board.h"
#include "Move.h"
#include "MoveOrderer.h"
#include "Tuning.h"
#include "Verification.h"
#include "uci.h"

#include <iomanip>

using namespace std;
using namespace bb;
using namespace move;

void main_tune_features_bb() {
    bb_init();
    Evaluator* evaluator = new Evaluator();

    using namespace tuning;

    loadPositionFile("resources/other/quiet-labeled.epd", 1e7);
    auto K = tuning::computeK(evaluator, 2.86681, 200, 1e-7);

    // tune Phase specificly
    float* params     = evaluator->getPhaseValues();
    int    paramCount = 16;

    for (int i = 0; i < 5000; i++) {

        std::cout << "--------------------------------------------------- [" << i
                  << "] ----------------------------------------------" << std::endl;

        std::cout << std::setprecision(8) << tuning::optimiseBlackBox(evaluator, K, params, paramCount, 0.3)
                  << std::endl;

        for (int e = 0; e < paramCount; e++) {
            std::cout << std::setw(14) << evaluator->getPhaseValues()[e] << ",";
        }

        std::cout << std::endl;
    }

    delete evaluator;
    bb_cleanUp();
}

void main_tune_pst_bb(Piece piece) {
    bb_init();
    Evaluator* evaluator = new Evaluator();

    using namespace tuning;

    loadPositionFile("resources/other/quiet-labeled.epd", 1e7);
    auto K = tuning::computeK(evaluator, 2.86681, 200, 1e-7);

    for (int i = 0; i < 64; i++) {
        evaluator->getPSQT(piece, true)[i]  = round(evaluator->getPSQT(piece, true)[i]);
        evaluator->getPSQT(piece, false)[i] = round(evaluator->getPSQT(piece, false)[i]);
    }

    for (int i = 0; i < 5000; i++) {

        std::cout << "--------------------------------------------------- [" << i
                  << "] ----------------------------------------------" << std::endl;

        std::cout << std::setprecision(8)
                  << tuning::optimiseBlackBox(evaluator, K, evaluator->getPSQT(piece, true), 64, 1) << std::endl;
        std::cout << std::setprecision(8)
                  << tuning::optimiseBlackBox(evaluator, K, evaluator->getPSQT(piece, false), 64, 1) << std::endl;

        for (int n = 0; n < 64; n++) {
            std::cout << std::right << std::setw(6) << evaluator->getPSQT(piece, true)[n] << ",";
            if (n % 8 == 7)
                std::cout << std::endl;
        }
        for (int n = 0; n < 64; n++) {
            std::cout << std::right << std::setw(6) << evaluator->getPSQT(piece, false)[n] << ",";
            if (n % 8 == 7)
                std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    delete evaluator;
    bb_cleanUp();
}

void main_tune_features() {
    bb_init();
    Evaluator* evaluator = new Evaluator();

    using namespace tuning;

    loadPositionFile("resources/other/quiet-labeled.epd", 1e6);
    auto K = tuning::computeK(evaluator, 2.86681, 200, 1e-7);

    for (int i = 0; i < 5000; i++) {

        std::cout << "--------------------------------------------------- [" << i
                  << "] ----------------------------------------------" << std::endl;
        std::cout << std::setprecision(8) << tuning::optimiseGD(evaluator, K, 1e4) << std::endl;

        for (int k = 0; k < evaluator->paramCount(); k++) {
            std::cout << std::setw(14) << evaluator->getEarlyGameParams()[k] << ",";
        }
        std::cout << std::endl;
        for (int k = 0; k < evaluator->paramCount(); k++) {
            std::cout << std::setw(14) << evaluator->getLateGameParams()[k] << ",";
        }
        std::cout << std::endl;
    }

    delete evaluator;
    bb_cleanUp();
}

#ifdef TUNE_PST
void main_tune_pst() {
    bb_init();
    Evaluator* evaluator = new Evaluator();

    using namespace tuning;

    loadPositionFile("resources/quiet-labeled.epd", 1e6);

    // auto K = tuning::computeK(evaluator,2.86681, 200, 1e-7);

    for (int i = 0; i < 5000; i++) {

        std::cout << "--------------------------------------------------- [" << i
                  << "] ----------------------------------------------" << std::endl;

        std::cout << std::setprecision(8) << tuning::optimisePST(evaluator, 2.86681, 1e6) << std::endl;

        for (int k = 0; k < 64; k++) {
            std::cout << std::setprecision(1) << fixed << std::setw(10) << evaluator->getTunablePST_MG()[k] << ",";
            if (k % 8 == 7)
                std::cout << "\n";
        }
        std::cout << std::endl;
        for (int k = 0; k < 64; k++) {
            std::cout << std::setprecision(1) << fixed << std::setw(10) << evaluator->getTunablePST_EG()[k] << ",";
            if (k % 8 == 7)
                std::cout << "\n";
        }
        std::cout << std::endl;
    }

    delete evaluator;
    bb_cleanUp();
}
#endif

int main(int argc, char* argv[]) {

    if (argc == 1) {
        uci_loop(false);
    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        uci_loop(true);
    }


    /**********************************************************************************
     *                                  T U N I N G                                   *
     **********************************************************************************/
    
//    float psqt_king[64] = {
//        -96,12,117,-12,-58,0,48,17,
//        72,61,24,128,17,0,-6,-120,
//        24,80,48,-24,63,81,120,-17,
//        0,12,-24,-2,-8,-17,0,-71,
//        -61,24,-32,-74,-79,-47,-30,-83,
//        -4,8,-15,-43,-51,-17,10,-37,
//        16,23,-9,-43,-37,-17,31,33,
//        -17,40,21,-55,-5,-29,47,33,
//    };
//    float psqt_king_endgame[64] = {
//        -104,-48,-48,-32,-11,4,-12,-28,
//        -33,0,10,-6,8,35,14,25,
//        -10,0,11,18,2,30,20,-1,
//        -28,6,24,25,20,28,12,-2,
//        -23,-18,23,37,38,24,2,-11,
//        -31,-9,15,29,33,21,2,-12,
//        -49,-19,10,20,20,7,-15,-39,
//        -79,-55,-32,-5,-25,-11,-46,-79,
//    };
//
//
//    std::cout << "EvalScore psqt_king[64]{" << std::endl << "   ";
//    for(int i = 0; i < 64; i++){
//
//        std::cout << "M(" << setw(4) << psqt_king[i] << "," << setw(4) << psqt_king_endgame[i] << "), ";
//
//        if(i % 8 == 7){
//            std::cout << std::endl;
//            if(i / 8 < 7){
//                std::cout << "   ";
//            }
//        }
//    }
//    std::cout << "};";
    
    
    
    
    
    //    bb_init();
    //    tuning::loadPositionFile("../resources/other/quiet-labeled.epd", 750000);
    //    double K = tuning::computeK(new Evaluator(), 3, 100, 1e-6);
    //
    //    Evaluator ev {};
    //
    //    Piece p = PAWN;
    //
    //    for (int i = 0; i < 200; i++) {
    //        for (int p = PAWN; p <= KING; p++) {
    //            std::cout << tuning::optimiseBlackBox(&ev, K, ev.getPSQT(p, true), 64, 5-i/50) << std::endl;
    //            std::cout << tuning::optimiseBlackBox(&ev, K, ev.getPSQT(p, false), 64, 5-i/50) << std::endl;
    //
    //            std::cout << p << std::endl;
    //            for (int n = 0; n < 64; n++) {
    //                std::cout << ev.getPSQT(p, true)[n] << ",";
    //                if (n % 8 == 7) {
    //                    std::cout << std::endl;
    //                }
    //            }
    //            std::cout << std::endl;
    //
    //            for (int n = 0; n < 64; n++) {
    //                std::cout << ev.getPSQT(p, false)[n] << ",";
    //                if (n % 8 == 7) {
    //                    std::cout << std::endl;
    //                }
    //            }
    //
    //            std::cout << std::endl;
    //        }
    //    }
    //
    //    // main_tune_pst_bb(PAWN);
    //
    //    // main_tune_features();
    //    // main_tune_pst();
    //    // main_tune_features_bb();

    return 0;
}
