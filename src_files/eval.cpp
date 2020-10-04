//
// Created by finne on 5/31/2020.
//

#include "eval.h"

#include <immintrin.h>
#include <iomanip>

#define pst_index_white(s) squareIndex(7 - rankIndex(s), fileIndex(s))
#define pst_index_black(s) s

#define pawn_pst_index_white(square, wKSide)                                                                           \
    squareIndex(rankIndex(square), (wKSide ? fileIndex(square) : 7 - fileIndex(square)))
#define pawn_pst_index_black(square, bKSide)                                                                           \
    squareIndex(7 - rankIndex(square), (bKSide ? fileIndex(square) : 7 - fileIndex(square)))

float sqrts[28] = {
    0,       1,       1.41421, 1.73205, 2,       2.23607, 2.44949, 2.64575, 2.82843, 3,
    3.16228, 3.31662, 3.4641,  3.60555, 3.74166, 3.87298, 4,       4.12311, 4.24264, 4.3589,
    4.47214, 4.58258, 4.69042, 4.79583, 4.89898, 5,       5.09902, 5.19615,
};

float passer_rank[16] = {
    0, -18.000015, -14,        2.3,        18.599987, 23.999985,  16.299986,  0,
    0, 0.99999994, -41.700001, -16.700014, -14,       -29.000015, -31.700001, 0,
};

EvalScore psqt_pawn[64] {
    M(0, 0),    M(0, 0),    M(0, 0),     M(0, 0),     M(0, 0),     M(0, 0),    M(0, 0),     M(0, 0),
    M(-3, 18),  M(50, -4),  M(36, 9),    M(-11, 3),   M(-13, 15),  M(-30, 3),  M(-4, -14),  M(-23, -7),
    M(0, 5),    M(20, 5),   M(19, -12),  M(-4, -3),   M(-16, 2),   M(-3, -7),  M(-41, -14), M(-26, -8),
    M(-22, 24), M(-2, 11),  M(22, -9),   M(24, -18),  M(17, -12),  M(-5, -11), M(-25, 0),   M(-34, 4),
    M(-14, 41), M(11, 28),  M(30, 4),    M(31, -19),  M(26, -10),  M(8, 1),    M(-12, 18),  M(-18, 20),
    M(5, 111),  M(74, 109), M(98, 83),   M(84, 49),   M(34, 39),   M(20, 49),  M(0, 82),    M(4, 93),
    M(36, 247), M(42, 219), M(163, 189), M(127, 163), M(109, 172), M(81, 163), M(143, 214), M(48, 238),
    M(0, 0),    M(0, 0),    M(0, 0),     M(0, 0),     M(0, 0),     M(0, 0),    M(0, 0),     M(0, 0),
};

EvalScore psqt_knight[64] {
    M(-243, -48), M(-135, -43), M(-107, -5), M(-107, -31), M(18, -41), M(-220, -15), M(-82, -70), M(-174, -97),
    M(-136, -21), M(-98, -2),   M(65, -47),  M(-16, 3),    M(-30, -7), M(24, -38),   M(-49, -19), M(-96, -48),
    M(-96, -27),  M(22, -28),   M(-14, 7),   M(17, 0),     M(66, -24), M(104, -26),  M(20, -29),  M(0, -54),
    M(-31, -16),  M(-14, -7),   M(-31, 21),  M(23, 16),    M(-4, 18),  M(46, 0),     M(-19, 1),   M(5, -26),
    M(-31, -15),  M(-15, -15),  M(-15, 9),   M(-13, 21),   M(12, 12),  M(0, 6),      M(9, -4),    M(-17, -24),
    M(-45, -28),  M(-37, -2),   M(-6, -16),  M(-9, 11),    M(7, 1),    M(9, -15),    M(12, -36),  M(-32, -23),
    M(-32, -40),  M(-64, -21),  M(-28, -12), M(10, -19),   M(9, -11),  M(7, -27),    M(-10, -29), M(-20, -54),
    M(-144, -2),  M(-10, -47),  M(-59, -23), M(-38, -1),   M(7, -27),  M(-22, -20),  M(-6, -59),  M(-27, -75),
};

EvalScore psqt_bishop[64] {
    M(-103, 8),  M(-93, -10), M(-209, 28), M(-143, -8), M(-103, 14), M(-150, 15), M(-106, 18), M(-56, -9),
    M(-97, 3),   M(-37, 0),   M(-70, 1),   M(-119, 16), M(-32, 12),  M(15, -5),   M(-46, 3),   M(-161, 29),
    M(-79, 11),  M(-38, 2),   M(-27, 2),   M(-28, -8),  M(-33, 6),   M(-14, 1),   M(-20, 19),  M(-68, 7),
    M(-54, -4),  M(-43, 3),   M(-35, 10),  M(17, -5),   M(-11, 5),   M(-10, 3),   M(-49, 2),   M(-48, 8),
    M(-46, -11), M(-35, 0),   M(-27, 8),   M(-15, 7),   M(-3, 5),    M(-43, 12),  M(-33, -1),  M(-29, -2),
    M(-42, 3),   M(-18, 1),   M(-22, 11),  M(-24, 10),  M(-27, 15),  M(9, -13),   M(-22, 3),   M(-24, -2),
    M(-23, -3),  M(-21, -14), M(-20, -4),  M(-40, 6),   M(-19, 4),   M(-23, -1),  M(-2, -11),  M(-34, -30),
    M(-85, -3),  M(-20, 7),   M(-46, 2),   M(-41, 4),   M(-42, 8),   M(-44, 7),   M(-80, 11),  M(-68, -9),
};

EvalScore psqt_rook[64] {
    M(20, 20),  M(19, 14),  M(8, 24),   M(24, 17), M(45, 25),  M(12, 21), M(12, 21),  M(12, 15),
    M(29, 29),  M(34, 29),  M(64, 24),  M(58, 25), M(44, 11),  M(68, 20), M(36, 30),  M(24, 20),
    M(-3, 15),  M(27, 16),  M(21, 15),  M(36, 21), M(5, 16),   M(38, 13), M(56, 9),   M(-4, -3),
    M(-11, 17), M(-11, 10), M(25, 24),  M(21, 8),  M(20, 15),  M(47, 15), M(-15, 9),  M(-6, 6),
    M(-23, 14), M(-13, 16), M(3, 18),   M(-2, 13), M(8, 3),    M(-3, 7),  M(2, 5),    M(-17, -7),
    M(-37, 10), M(-13, 7),  M(-17, -1), M(-20, 7), M(-10, -4), M(1, -3),  M(-3, -2),  M(-34, -11),
    M(-33, 6),  M(-17, -1), M(-20, 9),  M(-1, 10), M(3, -5),   M(8, -5),  M(-21, -4), M(-64, 8),
    M(-10, -4), M(0, 8),    M(9, -1),   M(12, 5),  M(17, -3),  M(14, -2), M(-26, 6),  M(-11, -25),
};

EvalScore psqt_queen[64] {
    M(-50, 0),  M(-61, 49),  M(-42, 24),  M(-81, 27), M(45, -40),  M(17, -19), M(-8, -4),  M(32, 0),
    M(-51, 9),  M(-71, 24),  M(-46, 24),  M(-32, 15), M(-111, 61), M(23, -25), M(-9, 10),  M(24, 0),
    M(-33, 15), M(-33, 6),   M(6, -40),   M(-54, 36), M(-10, 16),  M(8, -12),  M(6, -6),   M(6, -13),
    M(-61, 57), M(-46, 30),  M(-40, 4),   M(-44, 13), M(-10, 0),   M(-13, 0),  M(-25, 44), M(-16, 26),
    M(-18, 9),  M(-58, 46),  M(-12, -12), M(-17, 18), M(-7, -2),   M(-3, -3),  M(-3, 25),  M(-2, 8),
    M(-37, 30), M(2, -38),   M(-17, 4),   M(-5, -18), M(-5, -9),   M(6, 2),    M(17, 13),  M(6, 24),
    M(-48, 21), M(-12, -16), M(5, -21),   M(9, -25),  M(23, -25),  M(18, -25), M(-1, -44), M(6, -27),
    M(-15, 11), M(-10, -12), M(-11, 4),   M(2, 5),    M(-8, 17),   M(-31, 13), M(-31, -4), M(-56, -25),
};

EvalScore psqt_king[64] {
    M(-96, -104), M(12, -48), M(117, -48), M(-12, -32), M(-58, -11), M(0, 4),     M(48, -12), M(17, -28),
    M(72, -33),   M(61, 0),   M(24, 10),   M(128, -6),  M(17, 8),    M(0, 35),    M(-6, 14),  M(-120, 25),
    M(24, -10),   M(80, 0),   M(48, 11),   M(-24, 18),  M(63, 2),    M(81, 30),   M(120, 20), M(-17, -1),
    M(0, -28),    M(12, 6),   M(-24, 24),  M(-2, 25),   M(-8, 20),   M(-17, 28),  M(0, 12),   M(-71, -2),
    M(-61, -23),  M(24, -18), M(-32, 23),  M(-74, 37),  M(-79, 38),  M(-47, 24),  M(-30, 2),  M(-83, -11),
    M(-4, -31),   M(8, -9),   M(-15, 15),  M(-43, 29),  M(-51, 33),  M(-17, 21),  M(10, 2),   M(-37, -12),
    M(16, -49),   M(23, -19), M(-9, 10),   M(-43, 20),  M(-37, 20),  M(-17, 7),   M(31, -15), M(33, -39),
    M(-17, -79),  M(40, -55), M(21, -32),  M(-55, -5),  M(-5, -25),  M(-29, -11), M(47, -46), M(33, -79),
};

EvalScore* psqt[6] {psqt_pawn, psqt_knight, psqt_bishop, psqt_rook, psqt_queen, psqt_king};

#ifdef TUNE_PST
float* tunablePST_MG_grad = new float[64] {};
float* tunablePST_EG_grad = new float[64] {};
#endif

float* _pieceValuesEarly = new float[unusedVariable] {
    90.211487,  69.899239,  6.2228494,  2.967576,   -9.1678696, 8.7445889,  -4.0076895, -11.351655, 14.176047,
    463.04376,  47.557571,  26.782915,  22.808609,  473.91107,  31.915777,  24.075211,  41.845776,  -5.8912878,
    13.943347,  574.08636,  101.79205,  22.015999,  61.792591,  17.496267,  12.106392,  1350.0123,  49.171928,
    8.0235796,  378.32214,  253.04475,  -51.211845, 10.589941,  -6.3968649, 6.3443727,  19.223135,  -13.720773,
    0.25771901, -40.303791, -41.210938, -35.532829, -19.787298, -68.714424, 9.4156246,  -62.58881,  -14.393072,
    -10.375373, -183.67802, -316.65826, -12.65072,  -7.1560545, -18.819189, -10.803145, -12.076631, -10.146332,
    11.989947,  -6.6928525, 4.9275126,  0,          0,          0,
};

float* _pieceValuesLate = new float[unusedVariable] {
    104.82733,  130.21487,  11.598015,  41.836597,  -7.8446722, -4.1840973, -15.864173, 0.44860947, -6.6336184,
    327.79779,  81.411537,  16.900436,  17.016239,  287.31894,  8.0259972,  30.608137,  51.638828,  5.4482141,
    18.636389,  590.33789,  100.65056,  27.68157,   -17.710117, 1.9092197,  6.6818852,  1121.1086,  27.764854,
    57.6642,    -129.18095, 41.508553,  52.540432,  1.0327644,  -1.8575548, -28.430937, -38.164093, 24.079365,
    30.723057,  -144.93318, -50.13253,  -11.376001, -8.0586424, -61.548996, -81.147713, -381.26257, -7.7224793,
    -74.114044, -515.90515, -630.17755, -16.433153, 2.7820337,  -2.0064692, -7.3157744, -5.1838088, -8.3675108,
    -3.3051472, -11.926251, 16.130898,  0,          0,          0,
};

float* phaseValues = new float[6] {
    0, 1, 1, 2, 4, 0,
};

// TODO tweak values
float kingSafetyTable[100] {0,   0,   1,   2,   3,   5,   7,   9,   12,  15,  18,  22,  26,  30,  35,  39,  44,
                            50,  56,  62,  68,  75,  82,  85,  89,  97,  105, 113, 122, 131, 140, 150, 169, 180,
                            191, 202, 213, 225, 237, 248, 260, 272, 283, 295, 307, 319, 330, 342, 354, 366, 377,
                            389, 401, 412, 424, 436, 448, 459, 471, 483, 494, 500, 500, 500, 500, 500, 500, 500,
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
                            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500};

/**
 * adds the factor to value of attacks if the piece attacks the kingzone
 * @param attacks
 * @param kingZone
 * @param pieceCount
 * @param valueOfAttacks
 * @param factor
 */
void addToKingSafety(U64 attacks, U64 kingZone, int& pieceCount, int& valueOfAttacks, int factor) {
    if (attacks & kingZone) {
        pieceCount++;
        valueOfAttacks += factor * bitCount(attacks & kingZone);
    }
}

/**
 * checks if the given square is an outpost given the color and a bitboard of the opponent pawns
 */
bool isOutpost(Square s, Color c, U64 opponentPawns, U64 pawnCover) {
    U64 sq = ONE << s;

    if (c == WHITE) {
        if (((whitePassedPawnMask[s] & ~FILES[fileIndex(s)]) & opponentPawns) == 0 && (sq & pawnCover)) {
            return true;
        }
    } else {
        if (((blackPassedPawnMask[s] & ~FILES[fileIndex(s)]) & opponentPawns) == 0 && (sq & pawnCover)) {
            return true;
        }
    }
    return false;
}

void Evaluator::computeHangingPieces(Board* b) {
    U64 WnotAttacked = ~b->getAttackedSquares(WHITE);
    U64 BnotAttacked = ~b->getAttackedSquares(BLACK);

    for (int i = PAWN; i <= QUEEN; i++) {
        features[INDEX_PAWN_HANGING + i] =
            +bitCount(b->getPieces(WHITE, i) & WnotAttacked) - bitCount(b->getPieces(BLACK, i) & BnotAttacked);
    }
}

void Evaluator::computePinnedPieces(Board* b) {

    for (int i = 0; i < 15; i++) {
        features[INDEX_PINNED_PAWN_BY_BISHOP + i] = 0;
    }

    Square square;
    Square wkingSq = bitscanForward(b->getPieces(WHITE, KING));
    U64    pinner  = lookUpRookXRayAttack(wkingSq, *b->getOccupied(), b->getTeamOccupied()[WHITE])
                 & (b->getPieces(BLACK, ROOK) | b->getPieces(BLACK, QUEEN));
    while (pinner) {
        square             = bitscanForward(pinner);
        Square pinnedPlace = bitscanForward(inBetweenSquares[wkingSq][square] & b->getTeamOccupied()[WHITE]);
        features[INDEX_PINNED_PAWN_BY_BISHOP + 3 * (b->getPiece(pinnedPlace) % 6)
                 + (b->getPiece(square) % 6 - BISHOP)] += 1;
        pinner = lsbReset(pinner);
    }

    pinner = lookUpBishopXRayAttack(wkingSq, *b->getOccupied(), b->getTeamOccupied()[WHITE])
             & (b->getPieces(BLACK, BISHOP) | b->getPieces(BLACK, QUEEN));
    while (pinner) {
        square             = bitscanForward(pinner);
        Square pinnedPlace = bitscanForward(inBetweenSquares[wkingSq][square] & b->getTeamOccupied()[WHITE]);

        features[INDEX_PINNED_PAWN_BY_BISHOP + 3 * (b->getPiece(pinnedPlace) % 6)
                 + (b->getPiece(square) % 6 - BISHOP)] += 1;
        pinner = lsbReset(pinner);
    }

    Square bkingSq = bitscanForward(b->getPieces(BLACK, KING));
    pinner         = lookUpRookXRayAttack(bkingSq, *b->getOccupied(), b->getTeamOccupied()[BLACK])
             & (b->getPieces(WHITE, ROOK) | b->getPieces(WHITE, QUEEN));
    while (pinner) {
        square             = bitscanForward(pinner);
        Square pinnedPlace = bitscanForward(inBetweenSquares[bkingSq][square] & b->getTeamOccupied()[BLACK]);
        features[INDEX_PINNED_PAWN_BY_BISHOP + 3 * (b->getPiece(pinnedPlace) % 6)
                 + (b->getPiece(square) % 6 - BISHOP)] -= 1;
        pinner = lsbReset(pinner);
    }
    pinner = lookUpBishopXRayAttack(bkingSq, *b->getOccupied(), b->getTeamOccupied()[BLACK])
             & (b->getPieces(WHITE, BISHOP) | b->getPieces(WHITE, QUEEN));
    while (pinner) {
        square             = bitscanForward(pinner);
        Square pinnedPlace = bitscanForward(inBetweenSquares[bkingSq][square] & b->getTeamOccupied()[BLACK]);

        features[INDEX_PINNED_PAWN_BY_BISHOP + 3 * (b->getPiece(pinnedPlace) % 6)
                 + (b->getPiece(square) % 6 - BISHOP)] -= 1;
        pinner = lsbReset(pinner);
    }
}

/**
 * evaluates the board.
 * @param b
 * @return
 */
bb::Score Evaluator::evaluate(Board* b) {

    Score res = 0;

#ifdef TUNE_PST
    for (int i = 0; i < 64; i++) {
        tunablePST_MG_grad[i] = 0;
        tunablePST_EG_grad[i] = 0;
    }
#endif

    EvalScore pstSum = M(0, 0);

    memset(features, 0, unusedVariable * sizeof(float));

    U64 whiteTeam = b->getTeamOccupied()[WHITE];
    U64 blackTeam = b->getTeamOccupied()[BLACK];
    U64 occupied  = *b->getOccupied();

    Square whiteKingSquare = bitscanForward(b->getPieces()[WHITE_KING]);
    Square blackKingSquare = bitscanForward(b->getPieces()[BLACK_KING]);

    U64 whiteKingZone = KING_ATTACKS[whiteKingSquare];
    U64 blackKingZone = KING_ATTACKS[blackKingSquare];

    Square square;
    U64    attacks;
    U64    k;

    // clang-format off
    phase =
            (24.0f + phaseValues[5]
             - phaseValues[0] * bitCount(
                    b->getPieces()[WHITE_PAWN] |
                    b->getPieces()[BLACK_PAWN])
             - phaseValues[1] * bitCount(
                    b->getPieces()[WHITE_KNIGHT] |
                    b->getPieces()[BLACK_KNIGHT])
             - phaseValues[2] * bitCount(
                    b->getPieces()[WHITE_BISHOP] |
                    b->getPieces()[BLACK_BISHOP])
             - phaseValues[3] * bitCount(
                    b->getPieces()[WHITE_ROOK] |
                    b->getPieces()[BLACK_ROOK])
             - phaseValues[4] * bitCount(
                    b->getPieces()[WHITE_QUEEN] |
                    b->getPieces()[BLACK_QUEEN])) / 24.0f;
    
    
    if (phase > 1) phase = 1;
    if (phase < 0) phase = 0;
    
    
    //values to scale early/lategame weights
    float earlyWeightScalar = (1 - phase);
    float lateWeightScalar  = (phase);
    
    //the pst are multiples of 100
    float earlyPSTScalar = earlyWeightScalar;
    float latePSTScalar  = lateWeightScalar;
    
    
    int whitekingSafety_attackingPiecesCount = 0;
    int whitekingSafety_valueOfAttacks       = 0;
    
    int blackkingSafety_attackingPiecesCount = 0;
    int blackkingSafety_valueOfAttacks       = 0;
    /**********************************************************************************
     *                                  P A W N S                                     *
     **********************************************************************************/
    
    
    U64 whitePawns = b->getPieces()[WHITE_PAWN];
    U64 blackPawns = b->getPieces()[BLACK_PAWN];
    
    bool wKSide = (fileIndex(bitscanForward(b->getPieces()[WHITE_KING])) > 3 ? 0 : 1);
    bool bKSide = (fileIndex(bitscanForward(b->getPieces()[BLACK_KING])) > 3 ? 0 : 1);
    
    
    //all passed pawns for white/black
    U64 whitePassers = wPassedPawns(whitePawns, blackPawns);
    U64 blackPassers = bPassedPawns(blackPawns, whitePawns);
    
    //doubled pawns without the pawn least developed
    U64 whiteDoubledWithoutFirst = wFrontSpans(whitePawns) & whitePawns;
    U64 blackDoubledWithoutFirst = bFrontSpans(blackPawns) & blackPawns;
    
    //all doubled pawns
    U64 whiteDoubledPawns = whiteDoubledWithoutFirst | (wRearSpans(whiteDoubledWithoutFirst) & whitePawns);
    U64 blackDoubledPawns = blackDoubledWithoutFirst | (bRearSpans(blackDoubledWithoutFirst) & blackPawns);
    
    //all isolated pawns
    U64 whiteIsolatedPawns = whitePawns & ~(fillFile(shiftWest(whitePawns) | shiftEast(whitePawns)));
    U64 blackIsolatedPawns = blackPawns & ~(fillFile(shiftWest(blackPawns) | shiftEast(blackPawns)));
    
    U64 whiteBlockedPawns = shiftNorth(whitePawns)&(whiteTeam|blackTeam);
    U64 blackBlockedPawns = shiftSouth(blackPawns)&(whiteTeam|blackTeam);
    
    k = whitePawns;
    while (k) {
        square = bitscanForward(k);
    
        pstSum += psqt_pawn[pawn_pst_index_white(square, wKSide)];
 
        if (getBit(whitePassers,square)) features[INDEX_PASSER_RANK] += passer_rank[getBit(whiteBlockedPawns,square)*8+rankIndex(square)]/10;

        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_PAWN];
    while (k) {
        square = bitscanForward(k);
    
        pstSum -= psqt_pawn[pawn_pst_index_black(square, bKSide)];

        
        if (getBit(blackPassers,square)) features[INDEX_PASSER_RANK] -= passer_rank[getBit(blackBlockedPawns,square)*8+7-rankIndex(square)]/10;

        k = lsbReset(k);
    }
    
    
    U64 whitePawnEastCover = shiftNorthEast(whitePawns) & whitePawns;
    U64 whitePawnWestCover = shiftNorthWest(whitePawns) & whitePawns;
    U64 blackPawnEastCover = shiftSouthEast(blackPawns) & blackPawns;
    U64 blackPawnWestCover = shiftSouthWest(blackPawns) & blackPawns;
    
    U64 whitePawnCover = shiftNorthEast(whitePawns) | shiftNorthWest(whitePawns);
    U64 blackPawnCover = shiftSouthEast(blackPawns) | shiftSouthWest(blackPawns);
    
    features[INDEX_PAWN_DOUBLED_AND_ISOLATED] =
            +bitCount(whiteIsolatedPawns & whiteDoubledPawns)
            - bitCount(blackIsolatedPawns & blackDoubledPawns);
    features[INDEX_PAWN_DOUBLED]              =
            +bitCount(~whiteIsolatedPawns & whiteDoubledPawns)
            - bitCount(~blackIsolatedPawns & blackDoubledPawns);
    features[INDEX_PAWN_ISOLATED]             =
            +bitCount(whiteIsolatedPawns & ~whiteDoubledPawns)
            - bitCount(blackIsolatedPawns & ~blackDoubledPawns);
    features[INDEX_PAWN_PASSED]               =
            +bitCount(whitePassers)
            - bitCount(blackPassers);
    features[INDEX_PAWN_VALUE]                =
            +bitCount(b->getPieces()[WHITE_PAWN])
            - bitCount(b->getPieces()[BLACK_PAWN]);
    features[INDEX_PAWN_STRUCTURE]            =
            +bitCount(whitePawnEastCover)
            + bitCount(whitePawnWestCover)
            - bitCount(blackPawnEastCover)
            - bitCount(blackPawnWestCover);
    features[INDEX_PAWN_OPEN]                 =
            +bitCount(whitePawns & ~fillSouth(blackPawns))
            - bitCount(blackPawns & ~fillNorth(whitePawns));
    features[INDEX_PAWN_BACKWARD]             =
            +bitCount(fillSouth(~wAttackFrontSpans(whitePawns) & blackPawnCover) & whitePawns)
            - bitCount(fillNorth(~bAttackFrontSpans(blackPawns) & whitePawnCover) & blackPawns);
    features[INDEX_BLOCKED_PAWN]              = 
            +bitCount(whiteBlockedPawns)
            -bitCount(blackBlockedPawns);
    
    
    
    /*
     * only these squares are counted for mobility
     */
    U64 mobilitySquaresWhite = ~whiteTeam & ~(blackPawnCover);
    U64 mobilitySquaresBlack = ~blackTeam & ~(whitePawnCover);
    
    /**********************************************************************************
     *                                  K N I G H T S                                 *
     **********************************************************************************/

    
    
    k = b->getPieces()[WHITE_KNIGHT];
    while (k) {
        square  = bitscanForward(k);
        attacks = KNIGHT_ATTACKS[square];
    
        pstSum += psqt_knight[pst_index_white(square)];

//        res += psqt_knight[pst_index_white(square)] * earlyPSTScalar;
//        res += psqt_knight_endgame[pst_index_white(square)] * latePSTScalar;
    
    
        features[INDEX_KNIGHT_MOBILITY] += sqrts[bitCount(KNIGHT_ATTACKS[square] & mobilitySquaresWhite)];
        features[INDEX_KNIGHT_OUTPOST] += isOutpost(square, WHITE, blackPawns, whitePawnCover);
        features[INDEX_KNIGHT_DISTANCE_ENEMY_KING] += manhattanDistance(square, blackKingSquare);
        
        
        addToKingSafety(attacks, blackKingZone, blackkingSafety_attackingPiecesCount, blackkingSafety_valueOfAttacks,
                        2);
        
        
        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_KNIGHT];
    while (k) {
        square  = bitscanForward(k);
        attacks = KNIGHT_ATTACKS[square];
    
    
        pstSum -= psqt_knight[pst_index_black(square)];
//        res -= psqt_knight[pst_index_black(square)] * earlyPSTScalar;
//        res -= psqt_knight_endgame[pst_index_black(square)] * latePSTScalar;;
    
    
        features[INDEX_KNIGHT_MOBILITY] -= sqrts[bitCount(attacks & mobilitySquaresBlack)];
        features[INDEX_KNIGHT_OUTPOST] -= isOutpost(square, BLACK, whitePawns, blackPawnCover);
        features[INDEX_KNIGHT_DISTANCE_ENEMY_KING] -= manhattanDistance(square, whiteKingSquare);
        
        addToKingSafety(attacks, whiteKingZone, whitekingSafety_attackingPiecesCount, whitekingSafety_valueOfAttacks,
                        2);
        
        k = lsbReset(k);
    }
    features[INDEX_KNIGHT_VALUE] = (bitCount(b->getPieces()[WHITE_KNIGHT]) -
                                    bitCount(b->getPieces()[BLACK_KNIGHT]));
    /**********************************************************************************
     *                                  B I S H O P S                                 *
     **********************************************************************************/

    k = b->getPieces()[WHITE_BISHOP];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpBishopAttack(square, occupied);

#ifdef TUNE_PST
        tunablePST_MG_grad[pst_index_white(square)] += _pieceValuesEarly[INDEX_BISHOP_PSQT] * (1-phase) / 100;
        tunablePST_EG_grad[pst_index_white(square)] += _pieceValuesLate [INDEX_BISHOP_PSQT] * phase     / 100;
#endif
    
//        res += psqt_bishop[pst_index_white(square)] * earlyPSTScalar;
//        res += psqt_bishop_endgame[pst_index_white(square)] * (phase) / 100.0;
    
        pstSum += psqt_bishop[pst_index_white(square)];


        features[INDEX_BISHOP_MOBILITY] += sqrts[bitCount(attacks & mobilitySquaresWhite)];
        features[INDEX_BISHOP_PAWN_SAME_SQUARE] += bitCount(
                blackPawns & (((ONE << square) & WHITE_SQUARES) ? WHITE_SQUARES : BLACK_SQUARES));
    
    
        features[INDEX_BISHOP_FIANCHETTO] +=
                (square == G2 &&
                 whitePawns & ONE << F2 &&
                 whitePawns & ONE << H2 &&
                 whitePawns & (ONE << G3 | ONE << G4));
        features[INDEX_BISHOP_FIANCHETTO] +=
                (square == B2 &&
                 whitePawns & ONE << A2 &&
                 whitePawns & ONE << C2 &&
                 whitePawns & (ONE << B3 | ONE << B4));
        
        
        addToKingSafety(attacks, blackKingZone, blackkingSafety_attackingPiecesCount, blackkingSafety_valueOfAttacks,
                        2);
        
        
        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_BISHOP];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpBishopAttack(square, occupied);

#ifdef TUNE_PST
        tunablePST_MG_grad[pst_index_black(square)] -= _pieceValuesEarly[INDEX_BISHOP_PSQT] * (1-phase) / 100;
        tunablePST_EG_grad[pst_index_black(square)] -= _pieceValuesLate [INDEX_BISHOP_PSQT] * phase     / 100;
#endif
        pstSum -= psqt_bishop[pst_index_black(square)];

//        res -= psqt_bishop[pst_index_black(square)] * earlyPSTScalar;
//        res -= psqt_bishop_endgame[pst_index_black(square)] * latePSTScalar;
    
        features[INDEX_BISHOP_MOBILITY] -= sqrts[bitCount(attacks & mobilitySquaresBlack)];
        features[INDEX_BISHOP_PAWN_SAME_SQUARE] -= bitCount(
                whitePawns & (((ONE << square) & WHITE_SQUARES) ? WHITE_SQUARES : BLACK_SQUARES));
    
    
        features[INDEX_BISHOP_FIANCHETTO] -=
                (square == G7 &&
                 blackPawns & ONE << F7 &&
                 blackPawns & ONE << H7 &&
                 blackPawns & (ONE << G6 | ONE << G5));
        features[INDEX_BISHOP_FIANCHETTO] -=
                (square == B2 &&
                 blackPawns & ONE << A7 &&
                 blackPawns & ONE << C7 &&
                 blackPawns & (ONE << B6 | ONE << B5));
        
        addToKingSafety(attacks, whiteKingZone, whitekingSafety_attackingPiecesCount, whitekingSafety_valueOfAttacks,
                        2);
        
        k = lsbReset(k);
    }
    features[INDEX_BISHOP_VALUE] = (bitCount(b->getPieces()[WHITE_BISHOP]) - bitCount(b->getPieces()[BLACK_BISHOP]));
    features[INDEX_BISHOP_DOUBLED] =
            (bitCount(b->getPieces()[WHITE_BISHOP]) == 2) - (bitCount(b->getPieces()[BLACK_BISHOP]) == 2);
    /**********************************************************************************
     *                                  R O O K S                                     *
     **********************************************************************************/

    k = b->getPieces()[WHITE_ROOK];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpRookAttack(square, occupied);
    
    
//        res += psqt_rook[pst_index_white(square)] * earlyPSTScalar;
//        res += psqt_rook_endgame[pst_index_white(square)] * (phase) / 100.0;
    
        pstSum += psqt_rook[pst_index_white(square)];
    
        features[INDEX_ROOK_MOBILITY] += sqrts[bitCount(attacks & mobilitySquaresWhite)];
        
        if (lookUpRookAttack(square, ZERO) & b->getPieces()[BLACK_KING]) {
            //rook on same file or rank as king
            features[INDEX_ROOK_KING_LINE]++;
        }
        if ((whitePawns & FILES[fileIndex(square)]) == 0) {
            if ((blackPawns & FILES[fileIndex(square)]) == 0) {
                //open
                features[INDEX_ROOK_OPEN_FILE]++;
            } else {
                //half open
                features[INDEX_ROOK_HALF_OPEN_FILE]++;
            }
        }
        
        addToKingSafety(attacks, blackKingZone, blackkingSafety_attackingPiecesCount, blackkingSafety_valueOfAttacks,
                        3);
        
        
        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_ROOK];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpRookAttack(square, occupied);
    
        pstSum -= psqt_rook[pst_index_black(square)];

//        res -= psqt_rook[pst_index_black(square)] * earlyPSTScalar;
//        res -= psqt_rook_endgame[pst_index_black(square)] * (phase) / 100.0;
    
        features[INDEX_ROOK_MOBILITY] -= sqrts[bitCount(attacks & mobilitySquaresBlack)];
        
        if (lookUpRookAttack(square, ZERO) & b->getPieces()[WHITE_KING]) {
            //rook on same file or rank as king
            features[INDEX_ROOK_KING_LINE]--;
        }
        
        if ((whitePawns & FILES[fileIndex(square)]) == 0) {
            if ((blackPawns & FILES[fileIndex(square)]) == 0) {
                //open
                features[INDEX_ROOK_OPEN_FILE]--;
            } else {
                //half open
                features[INDEX_ROOK_HALF_OPEN_FILE]--;
            }
        }
        
        
        addToKingSafety(attacks, whiteKingZone, whitekingSafety_attackingPiecesCount, whitekingSafety_valueOfAttacks,
                        3);
        
        k = lsbReset(k);
    }
    features[INDEX_ROOK_VALUE] = (bitCount(b->getPieces()[WHITE_ROOK]) - bitCount(b->getPieces()[BLACK_ROOK]));
    
    /**********************************************************************************
     *                                  Q U E E N S                                   *
     **********************************************************************************/

    
    k = b->getPieces()[WHITE_QUEEN];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpRookAttack(square, occupied) | lookUpBishopAttack(square, occupied);
    
    
        pstSum += psqt_queen[pst_index_white(square)];
//        res += psqt_queen[pst_index_white(square)] * earlyPSTScalar;
//        res += psqt_queen_endgame[pst_index_white(square)] * latePSTScalar;
    
    
        features[INDEX_QUEEN_MOBILITY] += sqrts[bitCount(attacks & mobilitySquaresWhite)];
        features[INDEX_QUEEN_DISTANCE_ENEMY_KING] += manhattanDistance(square, blackKingSquare);
        
        addToKingSafety(attacks, blackKingZone, blackkingSafety_attackingPiecesCount, blackkingSafety_valueOfAttacks,
                        4);
        
        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_QUEEN];
    while (k) {
        square  = bitscanForward(k);
        attacks = lookUpRookAttack(square, occupied) | lookUpBishopAttack(square, occupied);
    
        pstSum -= psqt_queen[pst_index_black(square)];

//        res -= psqt_queen[pst_index_black(square)] * earlyPSTScalar;
//        res -= psqt_queen_endgame[pst_index_black(square)] * latePSTScalar;
    
    
        features[INDEX_QUEEN_MOBILITY] -= sqrts[bitCount(attacks & mobilitySquaresBlack)];
        features[INDEX_QUEEN_DISTANCE_ENEMY_KING] -= manhattanDistance(square, whiteKingSquare);
        
        addToKingSafety(attacks, whiteKingZone, whitekingSafety_attackingPiecesCount, whitekingSafety_valueOfAttacks,
                        4);
        
        k = lsbReset(k);
    }
    features[INDEX_QUEEN_VALUE] = bitCount(b->getPieces()[WHITE_QUEEN]) - bitCount(b->getPieces()[BLACK_QUEEN]);
    
    /**********************************************************************************
     *                                  K I N G S                                     *
     **********************************************************************************/
    k = b->getPieces()[WHITE_KING];

    
    while (k) {
        square = bitscanForward(k);
    
        pstSum += psqt_king[pst_index_white(square)];
//        res += psqt_king[pst_index_white(square)] * earlyPSTScalar;
//        res += psqt_king_endgame[pst_index_white(square)] * latePSTScalar;
    
        features[INDEX_KING_PAWN_SHIELD] += bitCount(KING_ATTACKS[square] & whitePawns);
        features[INDEX_KING_CLOSE_OPPONENT] += bitCount(KING_ATTACKS[square] & blackTeam);
        
        k = lsbReset(k);
    }
    
    k = b->getPieces()[BLACK_KING];
    while (k) {
        square = bitscanForward(k);

        pstSum -= psqt_king[pst_index_black(square)];

//        res -= psqt_king[pst_index_black(square)] * earlyPSTScalar;
//        res -= psqt_king_endgame[pst_index_black(square)] * latePSTScalar;
    
        features[INDEX_KING_PAWN_SHIELD] -= bitCount(KING_ATTACKS[square] & blackPawns);
        features[INDEX_KING_CLOSE_OPPONENT] -= bitCount(KING_ATTACKS[square] & whiteTeam);
        
        k = lsbReset(k);
    }
    
    computeHangingPieces(b);
    computePinnedPieces(b);
    
    features[INDEX_KING_SAFETY] =
            (kingSafetyTable[blackkingSafety_valueOfAttacks] - kingSafetyTable[whitekingSafety_valueOfAttacks]) / 100;
    features[INDEX_CASTLING_RIGHTS] =
            + b->getCastlingChance(STATUS_INDEX_WHITE_QUEENSIDE_CASTLING)
            + b->getCastlingChance(STATUS_INDEX_WHITE_KINGSIDE_CASTLING)
            - b->getCastlingChance(STATUS_INDEX_BLACK_QUEENSIDE_CASTLING)
            - b->getCastlingChance(STATUS_INDEX_BLACK_KINGSIDE_CASTLING);
    
    
    
    
    
    
    __m128 earlyRes{};
    __m128 lateRes{};
    
    for (int i = 0; i < unusedVariable; i += 4) {
        __m128 *feat = (__m128 *) (features + (i));
        
        __m128 *w1 = (__m128 *) (_pieceValuesEarly + (i));
        __m128 *w2 = (__m128 *) (_pieceValuesLate + (i));
        
        earlyRes = _mm_add_ps(earlyRes, _mm_mul_ps(*w1, *feat));
        lateRes  = _mm_add_ps(lateRes, _mm_mul_ps(*w2, *feat));
    }
    
    
    const __m128 tE   = _mm_add_ps(earlyRes, _mm_movehl_ps(earlyRes, earlyRes));
    const __m128 sumE = _mm_add_ss(tE, _mm_shuffle_ps(tE, tE, 1));
    const __m128 tL   = _mm_add_ps(lateRes, _mm_movehl_ps(lateRes, lateRes));
    const __m128 sumL = _mm_add_ss(tL, _mm_shuffle_ps(tL, tL, 1));
    
    res += phase * EgScore(pstSum) + (1-phase) * MgScore(pstSum);
    res += sumE[0] * (1 - phase) + sumL[0] * (phase);
    res += (b->getActivePlayer() == WHITE ? 15 : -15);
    return res;
    // clang-format on
}

void printEvaluation(Board* board) {

    using namespace std;

    Evaluator ev {};
    Score     score = ev.evaluate(board);
    float     phase = ev.getPhase();

    stringstream ss {};

    // String format = "%-30s | %-20s | %-20s %n";

    ss << std::setw(40) << std::left << "feature"
       << " | " << std::setw(20) << std::right << "difference"
       << " | " << std::setw(20) << "early weight"
       << " | " << std::setw(20) << "late weight"
       << " | " << std::setw(20) << "tapered weight"
       << " | " << std::setw(20) << "sum"
       << "\n";

    ss << "-----------------------------------------+----------------------+"
          "----------------------+----------------------+"
          "----------------------+----------------------+\n";
    ss << std::setw(40) << std::left << "PHASE"
       << " | " << std::setw(20) << std::right << ""
       << " | " << std::setw(20) << "0"
       << " | " << std::setw(20) << "1"
       << " | " << std::setw(20) << phase << " | " << std::setw(20) << phase << " | \n";

    ss << "-----------------------------------------+----------------------+"
          "----------------------+----------------------+"
          "----------------------+----------------------+\n";

    string names[] {
        "INDEX_PAWN_VALUE",
        "INDEX_PAWN_PSQT",
        "INDEX_PAWN_STRUCTURE",
        "INDEX_PAWN_PASSED",
        "INDEX_PAWN_ISOLATED",
        "INDEX_PAWN_DOUBLED",
        "INDEX_PAWN_DOUBLED_AND_ISOLATED",
        "INDEX_PAWN_BACKWARD",
        "INDEX_PAWN_OPEN",

        "INDEX_KNIGHT_VALUE",
        "INDEX_KNIGHT_PSQT",
        "INDEX_KNIGHT_MOBILITY",
        "INDEX_KNIGHT_OUTPOST",

        "INDEX_BISHOP_VALUE",
        "INDEX_BISHOP_PSQT",
        "INDEX_BISHOP_MOBILITY",
        "INDEX_BISHOP_DOUBLED",
        "INDEX_BISHOP_PAWN_SAME_SQUARE",
        "INDEX_BISHOP_FIANCHETTO",

        "INDEX_ROOK_VALUE",
        "INDEX_ROOK_PSQT",
        "INDEX_ROOK_MOBILITY",
        "INDEX_ROOK_OPEN_FILE",
        "INDEX_ROOK_HALF_OPEN_FILE",
        "INDEX_ROOK_KING_LINE",

        "INDEX_QUEEN_VALUE",
        "INDEX_QUEEN_PSQT",
        "INDEX_QUEEN_MOBILITY",

        "INDEX_KING_SAFETY",
        "INDEX_KING_PSQT",
        "INDEX_KING_CLOSE_OPPONENT",
        "INDEX_KING_PAWN_SHIELD",

        "INDEX_KNIGHT_DISTANCE_ENEMY_KING",
        "INDEX_QUEEN_DISTANCE_ENEMY_KING",

        "INDEX_PINNED_PAWN_BY_BISHOP",
        "INDEX_PINNED_PAWN_BY_ROOK",
        "INDEX_PINNED_PAWN_BY_QUEEN",
        "INDEX_PINNED_KNIGHT_BY_BISHOP",
        "INDEX_PINNED_KNIGHT_BY_ROOK",
        "INDEX_PINNED_KNIGHT_BY_QUEEN",
        "INDEX_PINNED_BISHOP_BY_BISHOP",
        "INDEX_PINNED_BISHOP_BY_ROOK",
        "INDEX_PINNED_BISHOP_BY_QUEEN",
        "INDEX_PINNED_ROOK_BY_BISHOP",
        "INDEX_PINNED_ROOK_BY_ROOK",
        "INDEX_PINNED_ROOK_BY_QUEEN",
        "INDEX_PINNED_QUEEN_BY_BISHOP",
        "INDEX_PINNED_QUEEN_BY_ROOK",
        "INDEX_PINNED_QUEEN_BY_QUEEN",

        "INDEX_PAWN_HANGING",
        "INDEX_KNIGHT_HANGING",
        "INDEX_BISHOP_HANGING",
        "INDEX_ROOK_HANGING",
        "INDEX_QUEEN_HANGING",

        // ignore this and place new values before here
        "-",
        "-",
        "-",
        "-",
    };

    for (int i = 0; i < unusedVariable; i++) {

        ss << std::setw(40) << std::left << names[i] << " | " << std::setw(20) << std::right << ev.getFeatures()[i]
           << " | " << std::setw(20) << ev.getEarlyGameParams()[i] << " | " << std::setw(20)
           << ev.getLateGameParams()[i] << " | " << std::setw(20)
           << ev.getEarlyGameParams()[i] * (1 - phase) + ev.getLateGameParams()[i] * phase << " | " << std::setw(20)
           << (ev.getEarlyGameParams()[i] * (1 - phase) + ev.getLateGameParams()[i] * phase) * ev.getFeatures()[i]
           << " | \n";
    }
    ss << "-----------------------------------------+----------------------+"
          "----------------------+----------------------+"
          "----------------------+----------------------+\n";

    ss << std::setw(40) << std::left << "TOTAL"
       << " | " << std::setw(20) << std::right << ""
       << " | " << std::setw(20) << ""
       << " | " << std::setw(20) << ""
       << " | " << std::setw(20) << ""
       << " | " << std::setw(20) << score << " | \n";

    ss << "-----------------------------------------+----------------------+"
          "----------------------+----------------------+"
          "----------------------+----------------------+\n";

    std::cout << ss.str() << std::endl;
}

float* Evaluator::getFeatures() { return features; }

float Evaluator::getPhase() { return phase; }

float* Evaluator::getEarlyGameParams() { return _pieceValuesEarly; }

float* Evaluator::getLateGameParams() { return _pieceValuesLate; }

int Evaluator::paramCount() { return unusedVariable; }

float* Evaluator::getPSQT(Piece piece, bool early) {
    switch (piece) {
        //        case PAWN: return early ? psqt_pawn : psqt_pawn_endgame;
        //        case KNIGHT: return early ? psqt_knight : psqt_knight_endgame;
        //        case BISHOP: return early ? psqt_bishop : psqt_bishop_endgame;
        //        case ROOK: return early ? psqt_rook : psqt_rook_endgame;
        //        case QUEEN: return early ? psqt_queen : psqt_queen_endgame;
        //        case KING: return early ? psqt_king : psqt_king_endgame;
    }
    return nullptr;
}
float* Evaluator::getPhaseValues() { return passer_rank; }
#ifdef TUNE_PST
float* Evaluator::getTunablePST_MG() { return psqt_bishop; }

float* Evaluator::getTunablePST_EG() { return psqt_bishop_endgame; }

float* Evaluator::getTunablePST_MG_grad() { return tunablePST_MG_grad; }

float* Evaluator::getTunablePST_EG_grad() { return tunablePST_EG_grad; }
#endif

void Material::reset(Board* b) {

    this->materialSum = M(0, 0);

    bool wKSide = (fileIndex(bitscanForward(b->getPieces()[WHITE_KING])) > 3 ? 0 : 1);
    bool bKSide = (fileIndex(bitscanForward(b->getPieces()[BLACK_KING])) > 3 ? 0 : 1);

    U64 k = b->getPieces(WHITE, PAWN);
    for (; k; k = lsbReset(k)) {
        Square square = bitscanForward(k);
        this->materialSum +=
            psqt_pawn[squareIndex(rankIndex(square), (wKSide ? fileIndex(square) : 7 - fileIndex(square)))];
    }
    k = b->getPieces(BLACK, PAWN);
    for (; k; k = lsbReset(k)) {
        Square square = bitscanForward(k);
        this->materialSum -=
            psqt_pawn[squareIndex(7 - rankIndex(square), (bKSide ? fileIndex(square) : 7 - fileIndex(square)))];
    }

    for (Piece p = KNIGHT; p <= KING; p++) {
        
        U64 k = b->getPieces(WHITE, p);
        for (; k; k = lsbReset(k)) {
            this->materialSum += psqt[p][pst_index_white(bitscanForward(k))];
        }
        k = b->getPieces(BLACK, p);
        for (; k; k = lsbReset(k)) {
            this->materialSum -= psqt[p][pst_index_black(bitscanForward(k))];
        }
    }
}

void Material::onMove(Board* b, Move m) {

    Square sqFrom    = getSquareFrom(m);
    Square sqTo      = getSquareTo(m);
    Piece  pFrom     = getMovingPiece(m);
    Piece  pFromPure = pFrom % 6;
    Type   mType     = getType(m);
    Color  color     = pFrom / 6;
    int    factor    = color == WHITE ? 1 : -1;

    bool wKSide = (fileIndex(bitscanForward(b->getPieces()[WHITE_KING])) > 3 ? 0 : 1);
    bool bKSide = (fileIndex(bitscanForward(b->getPieces()[BLACK_KING])) > 3 ? 0 : 1);

    if (color == WHITE) {
        if (pFromPure == PAWN) {

            if (isPromotion(m)) {

                this->materialSum -= psqt_pawn[pawn_pst_index_white(sqFrom, wKSide)];
                this->materialSum += psqt[promotionPiece(m) % 6][pst_index_white(sqTo)];

                if (isCapture(m)) {
                    // assuming we don't capture other pawns
                    this->materialSum += psqt[getCapturedPiece(m) % 6][pst_index_black(sqTo)];
                }
                return;
            } else if (mType == EN_PASSANT) {

                this->materialSum -= psqt_pawn[pawn_pst_index_white(sqFrom, wKSide)];
                this->materialSum += psqt_pawn[pawn_pst_index_white(sqTo, wKSide)];

                this->materialSum += psqt_pawn[pawn_pst_index_black(sqTo - 8 * factor, bKSide)];
                return;
            }

            this->materialSum -= psqt_pawn[pawn_pst_index_white(sqFrom, wKSide)];
            this->materialSum += psqt_pawn[pawn_pst_index_white(sqTo, wKSide)];
        } else if (pFrom % 6 == KING) {
            //no matter what type of move the king does, we can always move the piece
            this->materialSum -= psqt[KING][pst_index_white(sqFrom)];
            this->materialSum += psqt[KING][pst_index_white(sqTo)];
    
            //if the king crosses the half, we need to redo the pawn material
            if(fileIndex(sqTo) / 4 != fileIndex(sqFrom) / 4){
                reset(b);
                return;
            }
            
            //for a castling move, we also need to consider the rook. we can exit early as a promotion cannot be a capture.
            if (isCastle(m)) {
                Square rookSquare = sqFrom + (mType == QUEEN_CASTLE ? -4 : 3);
                Square rookTarget = sqTo + (mType == QUEEN_CASTLE ? 1 : -1);

                this->materialSum -= psqt[ROOK][pst_index_white(rookSquare)];
                this->materialSum += psqt[ROOK][pst_index_white(rookTarget)];

                // we dont need to consider captures here
                return;
            }
            
            
            
        } else {
            //doing the initial move for knights, bishops, rooks and queens
            this->materialSum -= psqt[pFromPure][pst_index_white(sqFrom)];
            this->materialSum += psqt[pFromPure][pst_index_white(sqTo)];
        }

        //for normal captures beside e.p. and promotions, remove the taken material.
        if (isCapture(m)) {
            if (getCapturedPiece(m) % 6 == PAWN) {
                this->materialSum += psqt_pawn[pawn_pst_index_black(sqTo, bKSide)];
            } else {
                this->materialSum += psqt[getCapturedPiece(m) % 6][pst_index_black(sqTo)];
            }
        }
        
    } else {
        if (pFromPure == PAWN) {
        
            if (isPromotion(m)) {
            
                this->materialSum += psqt_pawn[pawn_pst_index_black(sqFrom, bKSide)];
                this->materialSum -= psqt[promotionPiece(m) % 6][pst_index_black(sqTo)];
            
                if (isCapture(m)) {
                    // assuming we don't capture other pawns
                    this->materialSum -= psqt[getCapturedPiece(m) % 6][pst_index_white(sqTo)];
                }
                return;
            } else if (mType == EN_PASSANT) {
            
                this->materialSum += psqt_pawn[pawn_pst_index_black(sqFrom, bKSide)];
                this->materialSum -= psqt_pawn[pawn_pst_index_black(sqTo, bKSide)];
            
                this->materialSum -= psqt_pawn[pawn_pst_index_white(sqTo - 8 * factor, wKSide)];
                return;
            }
        
            this->materialSum += psqt_pawn[pawn_pst_index_black(sqFrom, bKSide)];
            this->materialSum -= psqt_pawn[pawn_pst_index_black(sqTo, bKSide)];
        } else if (pFrom % 6 == KING) {
            //no matter what type of move the king does, we can always move the piece
            this->materialSum += psqt[KING][pst_index_black(sqFrom)];
            this->materialSum -= psqt[KING][pst_index_black(sqTo)];
    
    
            if(fileIndex(sqTo) / 4 != fileIndex(sqFrom) / 4){
                reset(b);
                return;
            }
    
    
            //for a castling move, we also need to consider the rook. we can exit early as a promotion cannot be a capture.
            if (isCastle(m)) {
                Square rookSquare = sqFrom + (mType == QUEEN_CASTLE ? -4 : 3);
                Square rookTarget = sqTo + (mType == QUEEN_CASTLE ? 1 : -1);
            
                this->materialSum += psqt[ROOK][pst_index_black(rookSquare)];
                this->materialSum -= psqt[ROOK][pst_index_black(rookTarget)];
            
                // we dont need to consider captures here
                return;
            }
            
        } else {
            //doing the initial move for knights, bishops, rooks and queens
            this->materialSum += psqt[pFromPure][pst_index_black(sqFrom)];
            this->materialSum -= psqt[pFromPure][pst_index_black(sqTo)];
        }
    
        //for normal captures beside e.p. and promotions, remove the taken material.
        if (isCapture(m)) {
            if (getCapturedPiece(m) % 6 == PAWN) {
                this->materialSum -= psqt_pawn[pawn_pst_index_white(sqTo, wKSide)];
            } else {
                this->materialSum -= psqt[getCapturedPiece(m) % 6][pst_index_white(sqTo)];
            }
        }
    }
}
