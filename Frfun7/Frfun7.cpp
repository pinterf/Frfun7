#include "avisynth.h"
#include "Frfun7.h"

#include <math.h>
#include <algorithm>
#include "emmintrin.h"

#define FRFUN7_X86

#ifdef FRFUN7_X86
#include <emmintrin.h>
#endif

#ifdef _WIN32
#define AVS_FORCEINLINE __forceinline
#else
#define AVS_FORCEINLINE inline __attribute__((always_inline))
#endif


// SAD of 4x4 reference and 4x4 actual bytes
// return value is in sad
static void scalar_sad16(const uint8_t *ref, int ref_pitch, int offset, const uint8_t* rdst, int rdp_aka_pitch, int &sad)
{
  sad = 0;

  for (int x = 0; x < 4; x++) {
      int src0 = *(rdst + x + offset);
      int src1 = *(rdst + x + rdp_aka_pitch + offset);
      int src2 = *(rdst + x + rdp_aka_pitch * 2 + offset);
      int src3 = *(rdst + x + rdp_aka_pitch * 3 + offset);

      int ref0 = *(ref + x);
      int ref1 = *(ref + x + ref_pitch);
      int ref2 = *(ref + x + ref_pitch * 2);
      int ref3 = *(ref + x + ref_pitch * 3);

      sad += std::abs(src0 - ref0);
      sad += std::abs(src1 - ref1);
      sad += std::abs(src2 - ref2);
      sad += std::abs(src3 - ref3);
  }
}

// increments acc if rsad < threshold.
// sad: sad input from prev. function sad16 (=sad4x4)
// sad_acc: sad accumulator
static void scalar_comp(int &sad, int& sad_acc, int threshold)
{
//  const int one_if_inc = sad < threshold ? 1 : 0;
//  sad = one_if_inc ? 0xFFFFFFFF : 0; // output mask, used in acc4/acc16
//  sad_acc += one_if_inc;

  if (sad < threshold) {
      sad = 0xffffffff;
      sad_acc++;
  } else {
      sad = 0;
  }

  // output sad will get 0xFFFFFFFF or 0
  // original code:
  // 1 shr 31 sra 31 = 0xffffffff
  // 0 shr 31 sra 31 = 0x00000000
  // Need for next step for acc4/acc16
}

static void scalar_acc4(int mmA[4], const uint8_t *rdst, int &mask_by_sadcomp)
{
    if (mask_by_sadcomp) {
        for (int x = 0; x < 4; x++)
            mmA[x] += rdst[x]; // 8 bytes 4 words
    }
}

// fills mm4..mm7 with 4 words
static void scalar_acc16(int offset, const uint8_t *rdst, int rdp_aka_pitch, int& mask_by_sadcomp, int mm4[4], int mm5[4], int mm6[4], int mm7[4])
{
  scalar_acc4(mm4, rdst + offset, mask_by_sadcomp);
  scalar_acc4(mm5, rdst + 1 * rdp_aka_pitch + offset, mask_by_sadcomp);
  scalar_acc4(mm6, rdst + 2 * rdp_aka_pitch + offset, mask_by_sadcomp);
  scalar_acc4(mm7, rdst + 3 * rdp_aka_pitch + offset, mask_by_sadcomp);
}

static void scalar_check(const uint8_t *ref, int ref_pitch, int offset, const uint8_t* rdst, int rdp_aka_pitch, int& racc, int threshold, int mm4[4], int mm5[4], int mm6[4], int mm7[4])
{
  int sad;
  scalar_sad16(ref, ref_pitch, offset, rdst, rdp_aka_pitch, sad);
  scalar_comp(sad, racc, threshold);
  scalar_acc16(offset, rdst, rdp_aka_pitch, sad, mm4, mm5, mm6, mm7);
}

static void scalar_acheck(const uint8_t *ref, int ref_pitch, int offset, const uint8_t* rdst, int rdp_aka_pitch, int& racc, int threshold, int mm4[4], int mm5[4], int mm6[4], int mm7[4], int &edx_sad_summing)
{
  int sad;
  scalar_sad16(ref, ref_pitch, offset, rdst, rdp_aka_pitch, sad);
  edx_sad_summing += sad;
  scalar_comp(sad, racc, threshold);
  scalar_acc16(offset, rdst, rdp_aka_pitch, sad, mm4, mm5, mm6, mm7);
}

static void scalar_stor4(uint8_t* esi, int mmA_array[4], int multiplier)
{
  // ((((mmA << 2) * multiplier) >> 16 ) + 1) >> 1
    for (int x = 0; x < 4; x++) {
        int mmA = mmA_array[x];

        mmA = (mmA * multiplier) >> 14;
        mmA = mmA + 1;
        mmA = mmA >> 1;
        esi[x] = mmA;
    }
}

template<int R> // radius; 3 or 0 is used
static void frcore_filter_b4r0or2or3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  // convert to upper left corner of the radius
  ptra += -R * pitcha - R; // cpln(-3, -3) or cpln(0, 0)

  int weight_acc = 0;

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  int mm4[4] = { 0, 0, 0, 0 };
  int mm5[4] = { 0, 0, 0, 0 };
  int mm6[4] = { 0, 0, 0, 0 };
  int mm7[4] = { 0, 0, 0, 0 };

  if constexpr (R >= 2)
  {
    if constexpr (R >= 3)
    {
      // -3 // top line of y= -3..+3
      scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
      scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      ptra += pitcha; // next line
    }

    // -2
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }
    ptra += pitcha; // next line

    // -1
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }
    ptra += pitcha; // next line
  }

  //; 0
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 2)
  {
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    if constexpr (R >= 3)
    {
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    }
  }

  if constexpr (R >= 2)
  {

    ptra += pitcha;

    // +1
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

    ptra += pitcha;
    // +2
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

    if constexpr (R >= 3)
    {
      ptra += pitcha;
      scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
      scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  *weight = weight_acc;

  // scale 4 - 7 by weight
  int weight_recip = inv_table[weight_acc];

  scalar_stor4(ptrb + 0 * pitchb, mm4, weight_recip);
  scalar_stor4(ptrb + 1 * pitchb, mm5, weight_recip);
  scalar_stor4(ptrb + 2 * pitchb, mm6, weight_recip);
  scalar_stor4(ptrb + 3 * pitchb, mm7, weight_recip);
}

static void frcore_filter_b4r3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_scalar<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

static void frcore_filter_b4r2_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_scalar<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

static void frcore_filter_b4r0_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_scalar<0>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// R == 2 or 3 (initially was: only 3)
template<int R>
static void frcore_filter_adapt_b4r2or3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  // convert to upper left corner of the radius
  ptra += -1 * pitcha - R; // cpln(-3, -1)

  int weight_acc = 0; // xor ecx, ecx

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  int mm4[4] = { 0, 0, 0, 0 };
  int mm5[4] = { 0, 0, 0, 0 };
  int mm6[4] = { 0, 0, 0, 0 };
  int mm7[4] = { 0, 0, 0, 0 };

  int edx_sad_summing = 0;

  // ; -1
  scalar_acheck(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  ptra += pitcha; // next line

  // ; 0
  scalar_acheck(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  ptra += pitcha; // next line

  // ; +1
  scalar_acheck(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  scalar_acheck(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);

  if (edx_sad_summing >= sThresh2)
  {
    // Expand the search for distances not covered in the first pass
    ptra -= 3 * pitcha; // move to -2

    // ; -2
    scalar_acheck(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; -1
    scalar_acheck(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; 0
    scalar_acheck(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; +1
    scalar_acheck(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; +2
    scalar_acheck(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    scalar_acheck(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);

    if constexpr (R >= 3) {
      if (edx_sad_summing >= sThresh3)
      {
        // Expand the search for distances not covered in the first-second pass
        ptra -= 5 * pitcha; // move to -3

        // no more need for acheck.edx_sad_summing

        // ; -3
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; -2
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; -1
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; 0
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +1
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +2
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +3
        scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
      }
    }
  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  *weight = weight_acc;

  // scale 4 - 7 by weight
  int weight_recip = inv_table[weight_acc];

  scalar_stor4(ptrb + 0 * pitchb, mm4, weight_recip);
  scalar_stor4(ptrb + 1 * pitchb, mm5, weight_recip);
  scalar_stor4(ptrb + 2 * pitchb, mm6, weight_recip);
  scalar_stor4(ptrb + 3 * pitchb, mm7, weight_recip);
}

static void frcore_filter_adapt_b4r3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  frcore_filter_adapt_b4r2or3_scalar<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, sThresh2, sThresh3, inv_table, weight);
}

static void frcore_filter_adapt_b4r2_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  frcore_filter_adapt_b4r2or3_scalar<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, sThresh2, sThresh3, inv_table, weight);
}

static void scalar_blend_store4(uint8_t* esi, int mmA_array[4], int mm2_multiplier)
{
    for (int x = 0; x < 4; x++) {
        int mmA = mmA_array[x];
        int mm3 = esi[x];
        // tmp= ((esi << 6) * multiplier) >> 16  ( == [esi]/1024 * multiplier)
        // mmA = (mmA + tmp + rounder_16) / 32

        mm3 = (mm3 * mm2_multiplier) >> 10; // pmulhw, signed
        mmA = mmA + mm3;
        mmA = mmA + 16;
        mmA = mmA >> 5;
        esi[x] = mmA;
    }
}

// used in mode_temporal
// R is 2 or 3
template<int R>
static void frcore_filter_overlap_b4r2or3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  ptra += -R * pitcha - R; // cpln(-3, -3) or cpln(-2, -2)

  int weight_acc = 0;

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  int mm4[4] = { 0, 0, 0, 0 };
  int mm5[4] = { 0, 0, 0, 0 };
  int mm6[4] = { 0, 0, 0, 0 };
  int mm7[4] = { 0, 0, 0, 0 };

  if constexpr (R >= 3)
  {
    // -3 // top line of y= -3..+3
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    ptra += pitcha; // next line
  }
  // -2
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha; // next line

  // -1
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha; // next line

  //; 0
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha;

  // +1
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  ptra += pitcha;
  // +2
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  if constexpr (R >= 3)
  {
    ptra += pitcha;
    scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    scalar_check(ptrr, pitchr, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  // weight variable is a multi-purpose one, here we get a 32 bit value,
  // which is really two 16 bit words
  // lower 16 and upper 16 bit has separate meaning
  int prev_weight = *weight;

  // write back real weight, probably a later process need it
  *weight = weight_acc;

  // scale 4 - 7 by weight and store(here with blending)
  int weight_recip = inv_table[weight_acc];

  int weight_lo16 = prev_weight & 0xFFFF; // lower 16 bit

  for (int x = 0; x < 4; x++) {
      mm4[x] = (mm4[x] * weight_recip + 256) >> 9;
      mm5[x] = (mm5[x] * weight_recip + 256) >> 9;
      mm6[x] = (mm6[x] * weight_recip + 256) >> 9;
      mm7[x] = (mm7[x] * weight_recip + 256) >> 9;

      mm4[x] = (mm4[x] * weight_lo16) >> 16;
      mm5[x] = (mm5[x] * weight_lo16) >> 16;
      mm6[x] = (mm6[x] * weight_lo16) >> 16;
      mm7[x] = (mm7[x] * weight_lo16) >> 16;
  }

  int weight_hi16 = prev_weight >> 16; // upper 16 bit

  /*
    blend is >>5 inside then it would need rounder_16. and not 24 (16+8)
    // to be change to the good one after porting
    pcmpeqd mm1, mm1      1111111111111111
    psrlw	mm1, 14                       11
    psllw	mm1, 3                     11000 // 16+8? why not 16
  */

  scalar_blend_store4(ptrb + 0 * pitchb, mm4, weight_hi16);
  scalar_blend_store4(ptrb + 1 * pitchb, mm5, weight_hi16);
  scalar_blend_store4(ptrb + 2 * pitchb, mm6, weight_hi16);
  scalar_blend_store4(ptrb + 3 * pitchb, mm7, weight_hi16);
}

static void frcore_filter_overlap_b4r3_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_overlap_b4r2or3_scalar<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// used in adaptive overlapping
// bottleneck in P = 1
static void frcore_filter_overlap_b4r2_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_overlap_b4r2or3_scalar<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// mmA is input/output. In scalar_blend_store4 mmA in input only
static void scalar_blend_diff4(uint8_t* esi, int mmA[4], int mm2_multiplier)
{
    int mm3[4];

    for (int x = 0; x < 4; x++) {
        mm3[x] = esi[x];

        // tmp= ((esi << 6) * multiplier) >> 16  ( == [esi]/1024 * multiplier)
        // mmA = (mmA + tmp + rounder_16) / 32
        // ((((mm1 << 2) * multiplier) >> 16 ) + 1) >> 1

        mm3[x] = (mm3[x] * mm2_multiplier) >> 10; // pmulhw, signed
        mmA[x] = mmA[x] + mm3[x];
        mmA[x] = mmA[x] + 16;
        mmA[x] = mmA[x] >> 5;
        esi[x] = mmA[x];
    }

    // this is the only difference from scalar_blend_store4
    mmA[0] = std::abs(mmA[0] - mm3[0]);
    mmA[0] += std::abs(mmA[1] - mm3[1]);
    mmA[0] += std::abs(mmA[2] - mm3[2]);
    mmA[0] += std::abs(mmA[3] - mm3[3]);
}

static void frcore_filter_diff_b4r1_scalar(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{

  ptra += -1 * pitcha - 1; //  cpln(-1, -1)

  int weight_acc = 0;

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  int mm4[4] = { 0, 0, 0, 0 };
  int mm5[4] = { 0, 0, 0, 0 };
  int mm6[4] = { 0, 0, 0, 0 };
  int mm7[4] = { 0, 0, 0, 0 };

  // -1
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  ptra += pitcha; // next line

  // 0
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  ptra += pitcha; // next line

  // 0
  scalar_check(ptrr, pitchr, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  scalar_check(ptrr, pitchr, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);

  // mm4 - mm7 has accumulated sum, weight is ready here

  // weight variable is a multi-purpose one, here we get a 32 bit value,
  // which is really two 16 bit words
  // lower 16 and upper 16 bit has separate meaning
  int prev_weight = *weight;

  // scale 4 - 7 by weight and store(here with blending)
  int weight_recip = inv_table[weight_acc];

  int weight_lo16 = prev_weight & 0xFFFF; // lower 16 bit

  for (int x = 0; x < 4; x++) {
      mm4[x] = (mm4[x] * weight_recip + 256) >> 9;
      mm5[x] = (mm5[x] * weight_recip + 256) >> 9;
      mm6[x] = (mm6[x] * weight_recip + 256) >> 9;
      mm7[x] = (mm7[x] * weight_recip + 256) >> 9;

      mm4[x] = (mm4[x] * weight_lo16) >> 16;
      mm5[x] = (mm5[x] * weight_lo16) >> 16;
      mm6[x] = (mm6[x] * weight_lo16) >> 16;
      mm7[x] = (mm7[x] * weight_lo16) >> 16;
  }

  int weight_hi16 = prev_weight >> 16; // upper 16 bit

  /*
    blend is >>5 inside then it would need rounder_16. and not 24 (16+8)
    // to be change to the good one after porting
    pcmpeqd mm1, mm1      1111111111111111
    psrlw	mm1, 14                       11
    psllw	mm1, 3                     11000 // 16+8? why not 16
  */

  scalar_blend_diff4(ptrb + 0 * pitchb, mm4, weight_hi16);
  scalar_blend_diff4(ptrb + 1 * pitchb, mm5, weight_hi16);
  scalar_blend_diff4(ptrb + 2 * pitchb, mm6, weight_hi16);
  scalar_blend_diff4(ptrb + 3 * pitchb, mm7, weight_hi16);

  *weight = mm4[0] + mm5[0] + mm6[0] + mm7[0];
  // mm4, mm5, mm6, mm7 are changed, outputs are SAD
}

static void frcore_dev_b4_scalar(const uint8_t* ptra, int pitcha, int* dev)
{
  ptra += - 1; // cpln(-1, 0).ptr;

  int sad1;
  scalar_sad16(ptra + 1, pitcha, 0, ptra + pitcha, pitcha, sad1);

  int sad2;
  scalar_sad16(ptra + 1, pitcha, 2, ptra + pitcha, pitcha, sad2);

  *dev = std::min(sad1, sad2);
}

static void frcore_sad_b4_scalar(const uint8_t* ptra, int pitcha, const uint8_t* ptrb, int pitchb, int* sad)
{
  int sad1;
  scalar_sad16(ptra, pitcha, 0, ptrb, pitchb, sad1);

  *sad = sad1;
}


#ifdef FRFUN7_X86

AVS_FORCEINLINE __m128i _mm_load_si32(const uint8_t* ptr) {
  return _mm_castps_si128(_mm_load_ss((const float*)(ptr)));
}

// SAD of 2x(2x4) reference and 4x4 actual bytes
// return value is in sad
AVS_FORCEINLINE void simd_sad16(__m128i ref01, __m128i ref23, int offset, const uint8_t* rdst, int rdp_aka_pitch, int &sad)
{
  auto src0 = _mm_load_si32(rdst + offset);
  auto src1 = _mm_load_si32(rdst + rdp_aka_pitch + offset);
  auto src01 = _mm_or_si128(_mm_slli_epi64(src0, 32), src1); // make 8 side-by-side byte from 2x4
  auto sad01 = _mm_sad_epu8(src01, ref01); // sad against 0th and 1st line reference 4 pixels

  auto src2 = _mm_load_si32(rdst + rdp_aka_pitch * 2 + offset);
  auto src3 = _mm_load_si32(rdst + rdp_aka_pitch * 3 + offset);
  auto src23 = _mm_or_si128(_mm_slli_epi64(src2, 32), src3); // make 8 side-by-side byte from 2x4
  auto sad23 = _mm_sad_epu8(src23, ref23); // sad against 2nd and 3rd line reference 4 pixels

  sad = _mm_cvtsi128_si32(_mm_add_epi64(sad01, sad23)); // lower 64 is enough
}

// increments acc if rsad < threshold.
// sad: sad input from prev. function sad16 (=sad4x4)
// sad_acc: sad accumulator
AVS_FORCEINLINE void simd_comp(int &sad, int& sad_acc, int threshold)
{
  const int one_if_inc = sad < threshold ? 1 : 0;
  sad = one_if_inc ? 0xFFFFFFFF : 0; // output mask, used in acc4/acc16
  sad_acc += one_if_inc;
  // output sad will get 0xFFFFFFFF or 0
  // original code:
  // 1 shr 31 sra 31 = 0xffffffff
  // 0 shr 31 sra 31 = 0x00000000
  // Need for next step for acc4/acc16
}

AVS_FORCEINLINE void simd_acc4(__m128i &mmA, __m128i mm3, int &mask_by_sadcomp)
{
  // mmA: 4 input words
  // mm3: 4 bytes
  // mask_by_sadcomp: all 0 or all FF
  __m128i mm2 = _mm_cvtsi32_si128(mask_by_sadcomp); // 0x00000000 of 0xFFFFFFFF from 'comp'
  mm3 = _mm_and_si128(mm3, mm2); // mask
  auto zero = _mm_setzero_si128();
  mmA = _mm_add_epi16(mmA, _mm_unpacklo_epi8(mm3, zero)); // 8 bytes 4 words
}

// fills mm4..mm7 with 4 words
AVS_FORCEINLINE void simd_acc16(int offset, const uint8_t *rdst, int rdp_aka_pitch, int& mask_by_sadcomp, __m128i &mm4, __m128i& mm5, __m128i& mm6, __m128i& mm7)
{
  simd_acc4(mm4, _mm_load_si32(rdst + offset), mask_by_sadcomp);
  simd_acc4(mm5, _mm_load_si32(rdst + 1 * rdp_aka_pitch + offset), mask_by_sadcomp);
  simd_acc4(mm6, _mm_load_si32(rdst + 2 * rdp_aka_pitch + offset), mask_by_sadcomp);
  simd_acc4(mm7, _mm_load_si32(rdst + 3 * rdp_aka_pitch + offset), mask_by_sadcomp);
}

AVS_FORCEINLINE void simd_check(__m128i ref01, __m128i ref23, int offset, const uint8_t* rdst, int rdp_aka_pitch, int& racc, int threshold,
  __m128i& mm4, __m128i& mm5, __m128i& mm6, __m128i& mm7)
{
  int sad;
  simd_sad16(ref01, ref23, offset, rdst, rdp_aka_pitch, sad);
  simd_comp(sad, racc, threshold);
  simd_acc16(offset, rdst, rdp_aka_pitch, sad, mm4, mm5, mm6, mm7);
}

AVS_FORCEINLINE void simd_acheck(__m128i ref01, __m128i ref23, int offset, const uint8_t* rdst, int rdp_aka_pitch, int& racc, int threshold,
  __m128i& mm4, __m128i& mm5, __m128i& mm6, __m128i& mm7,
  int &edx_sad_summing)
{
  int sad;
  simd_sad16(ref01, ref23, offset, rdst, rdp_aka_pitch, sad);
  edx_sad_summing += sad;
  simd_comp(sad, racc, threshold);
  simd_acc16(offset, rdst, rdp_aka_pitch, sad, mm4, mm5, mm6, mm7);
}

AVS_FORCEINLINE void simd_stor4(uint8_t* esi, __m128i& mmA, __m128i mm0_multiplier, __m128i mm3_rounder, __m128i mm2_zero)
{
  // ((((mm1 << 2) * multiplier) >> 16 ) + 1) >> 1
  auto mm1 = mmA;
  mm1 = _mm_slli_epi16(mm1, 2);
  mm1 = _mm_mulhi_epu16(mm1, mm0_multiplier); // pmulhuw, really unsigned
  mm1 = _mm_adds_epu16(mm1, mm3_rounder);
  mm1 = _mm_srli_epi16(mm1, 1);
  mm1 = _mm_packus_epi16(mm1, mm2_zero); // 4 words to 4 bytes
  *(uint32_t*)(esi) = _mm_cvtsi128_si32(mm1);
}

template<int R> // radius; 3 or 0 is used
AVS_FORCEINLINE void frcore_filter_b4r0or2or3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  // convert to upper left corner of the radius
  ptra += -R * pitcha - R; // cpln(-3, -3) or cpln(0, 0)

  int weight_acc = 0;

  // reference pixels
  auto m0 = _mm_load_si32(ptrr); // 4 bytes
  auto m1 = _mm_load_si32(ptrr + pitchr * 1);
  auto m2 = _mm_load_si32(ptrr + pitchr * 2);
  auto m3 = _mm_load_si32(ptrr + pitchr * 3);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  auto mm4 = _mm_setzero_si128();
  auto mm5 = _mm_setzero_si128();
  auto mm6 = _mm_setzero_si128();
  auto mm7 = _mm_setzero_si128();

  if constexpr (R >= 2)
  {
    if constexpr (R >= 3)
    {
      // -3 // top line of y= -3..+3
      simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
      simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      ptra += pitcha; // next line
    }

    // -2
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }
    ptra += pitcha; // next line

    // -1
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }
    ptra += pitcha; // next line
  }

  //; 0
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 2)
  {
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    if constexpr (R >= 3)
    {
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    }
  }

  if constexpr (R >= 2)
  {

    ptra += pitcha;

    // +1
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

    ptra += pitcha;
    // +2
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
    if constexpr (R >= 3)
    {
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

    if constexpr (R >= 3)
    {
      ptra += pitcha;
      simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
      simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 0
      simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 1
      simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 2
      simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7); // base - 3
    }

  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  *weight = weight_acc;

  auto zero = _mm_setzero_si128(); // packer zero
  auto rounder_one = _mm_set1_epi16(1);

  // scale 4 - 7 by weight
  auto weight_recip = _mm_set1_epi16(inv_table[weight_acc]);

  simd_stor4(ptrb + 0 * pitchb, mm4, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 1 * pitchb, mm5, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 2 * pitchb, mm6, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 3 * pitchb, mm7, weight_recip, rounder_one, zero);
}

AVS_FORCEINLINE void frcore_filter_b4r3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_simd<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

AVS_FORCEINLINE void frcore_filter_b4r2_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_simd<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

AVS_FORCEINLINE void frcore_filter_b4r0_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_b4r0or2or3_simd<0>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// R == 2 or 3 (initially was: only 3)
template<int R>
AVS_FORCEINLINE void frcore_filter_adapt_b4r2or3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  // convert to upper left corner of the radius
  ptra += -1 * pitcha - R; // cpln(-3, -1)

  int weight_acc = 0; // xor ecx, ecx

  // reference pixels
  auto m0 = _mm_load_si32(ptrr); // 4 bytes
  auto m1 = _mm_load_si32(ptrr + pitchr * 1);
  auto m2 = _mm_load_si32(ptrr + pitchr * 2);
  auto m3 = _mm_load_si32(ptrr + pitchr * 3);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  auto mm4 = _mm_setzero_si128();
  auto mm5 = _mm_setzero_si128();
  auto mm6 = _mm_setzero_si128();
  auto mm7 = _mm_setzero_si128();

  int edx_sad_summing = 0;

  // ; -1
  simd_acheck(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  ptra += pitcha; // next line

  // ; 0
  simd_acheck(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  ptra += pitcha; // next line

  // ; +1
  simd_acheck(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
  simd_acheck(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);

  if (edx_sad_summing >= sThresh2)
  {
    // Expand the search for distances not covered in the first pass
    ptra -= 3 * pitcha; // move to -2

    // ; -2
    simd_acheck(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; -1
    simd_acheck(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; 0
    simd_acheck(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; +1
    simd_acheck(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    ptra += pitcha; // next line

    // ; +2
    simd_acheck(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);
    simd_acheck(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7, edx_sad_summing);

    if constexpr (R >= 3) {
      if (edx_sad_summing >= sThresh3)
      {
        // Expand the search for distances not covered in the first-second pass
        ptra -= 5 * pitcha; // move to -3

        // no more need for acheck.edx_sad_summing

        // ; -3
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; -2
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; -1
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; 0
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +1
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +2
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        ptra += pitcha; // next line

        // ; +3
        simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
        simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
      }
    }
  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  *weight = weight_acc;

  auto zero = _mm_setzero_si128(); // packer zero
  auto rounder_one = _mm_set1_epi16(1);

  // scale 4 - 7 by weight
  auto weight_recip = _mm_set1_epi16(inv_table[weight_acc]);

  simd_stor4(ptrb + 0 * pitchb, mm4, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 1 * pitchb, mm5, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 2 * pitchb, mm6, weight_recip, rounder_one, zero);
  simd_stor4(ptrb + 3 * pitchb, mm7, weight_recip, rounder_one, zero);
}

AVS_FORCEINLINE void frcore_filter_adapt_b4r3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  frcore_filter_adapt_b4r2or3_simd<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, sThresh2, sThresh3, inv_table, weight);
}

AVS_FORCEINLINE void frcore_filter_adapt_b4r2_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, int sThresh2, int sThresh3, const int* inv_table, int* weight)
{
  frcore_filter_adapt_b4r2or3_simd<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, sThresh2, sThresh3, inv_table, weight);
}

AVS_FORCEINLINE void simd_blend_store4(uint8_t* esi, __m128i mmA, __m128i mm2_multiplier, __m128i mm1_rounder, __m128i mm0_zero)
{
  auto mm3 = _mm_unpacklo_epi8(_mm_load_si32(esi), mm0_zero);
  // tmp= ((esi << 6) * multiplier) >> 16  ( == [esi]/1024 * multiplier)
  // mmA = (mmA + tmp + rounder_16) / 32
  // ((((mm1 << 2) * multiplier) >> 16 ) + 1) >> 1
  mm3 = _mm_slli_epi16(mm3, 6);
  mm3 = _mm_mulhi_epi16(mm3, mm2_multiplier); // pmulhw, signed
  mmA = _mm_adds_epu16(mmA, mm3);
  mmA = _mm_adds_epu16(mmA, mm1_rounder);
  mmA = _mm_srli_epi16(mmA, 5);
  mmA = _mm_packus_epi16(mmA, mm0_zero); // 4 words to 4 bytes
  *(uint32_t*)(esi) = _mm_cvtsi128_si32(mmA);
}

// used in mode_temporal
// R is 2 or 3
template<int R>
AVS_FORCEINLINE void frcore_filter_overlap_b4r2or3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  ptra += -R * pitcha - R; // cpln(-3, -3) or cpln(-2, -2)

  int weight_acc = 0;

  // reference pixels
  auto m0 = _mm_load_si32(ptrr); // 4 bytes
  auto m1 = _mm_load_si32(ptrr + pitchr * 1);
  auto m2 = _mm_load_si32(ptrr + pitchr * 2);
  auto m3 = _mm_load_si32(ptrr + pitchr * 3);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  auto mm4 = _mm_setzero_si128();
  auto mm5 = _mm_setzero_si128();
  auto mm6 = _mm_setzero_si128();
  auto mm7 = _mm_setzero_si128();

  if constexpr (R >= 3)
  {
    // -3 // top line of y= -3..+3
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    ptra += pitcha; // next line
  }
  // -2
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha; // next line

  // -1
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha; // next line

  //; 0
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }
  ptra += pitcha;

  // +1
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  ptra += pitcha;
  // +2
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  if constexpr (R >= 3)
  {
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  if constexpr (R >= 3)
  {
    ptra += pitcha;
    simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 3, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 4, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 5, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
    simd_check(ref01, ref23, 6, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  }

  // mm4 - mm7 has accumulated sum, weight is ready here

  // weight variable is a multi-purpose one, here we get a 32 bit value,
  // which is really two 16 bit words
  // lower 16 and upper 16 bit has separate meaning
  int prev_weight = *weight;

  // write back real weight, probably a later process need it
  *weight = weight_acc;

  // scale 4 - 7 by weight and store(here with blending)
  auto weight_recip = _mm_set1_epi32(inv_table[weight_acc] + (1 << 16));

  mm4 = _mm_unpacklo_epi16(mm4, _mm_set1_epi16(256));
  mm5 = _mm_unpacklo_epi16(mm5, _mm_set1_epi16(256));
  mm6 = _mm_unpacklo_epi16(mm6, _mm_set1_epi16(256));
  mm7 = _mm_unpacklo_epi16(mm7, _mm_set1_epi16(256));

  // We do this instead of pmulhw in order to avoid a loss of precision,
  // which would result in a green tint (lower pixel values).
  mm4 = _mm_madd_epi16(mm4, weight_recip);
  mm5 = _mm_madd_epi16(mm5, weight_recip);
  mm6 = _mm_madd_epi16(mm6, weight_recip);
  mm7 = _mm_madd_epi16(mm7, weight_recip);

  mm4 = _mm_srli_epi32(mm4, 9);
  mm5 = _mm_srli_epi32(mm5, 9);
  mm6 = _mm_srli_epi32(mm6, 9);
  mm7 = _mm_srli_epi32(mm7, 9);

  auto weight_lo16 = _mm_set1_epi32(prev_weight & 0xFFFF); // lower 16 bit

  mm4 = _mm_mulhi_epi16(mm4, weight_lo16);
  mm5 = _mm_mulhi_epi16(mm5, weight_lo16);
  mm6 = _mm_mulhi_epi16(mm6, weight_lo16);
  mm7 = _mm_mulhi_epi16(mm7, weight_lo16);

  // Experiments show that there is no need to add, then subtract the sign bit in this case.
  mm4 = _mm_packs_epi32(mm4, _mm_setzero_si128());
  mm5 = _mm_packs_epi32(mm5, _mm_setzero_si128());
  mm6 = _mm_packs_epi32(mm6, _mm_setzero_si128());
  mm7 = _mm_packs_epi32(mm7, _mm_setzero_si128());

  auto weight_hi16 = _mm_set1_epi16(prev_weight >> 16); // upper 16 bit

  auto zero = _mm_setzero_si128(); // packer zero mm0

  /*
    blend is >>5 inside then it would need rounder_16. and not 24 (16+8)
    // to be change to the good one after porting
    pcmpeqd mm1, mm1      1111111111111111
    psrlw	mm1, 14                       11
    psllw	mm1, 3                     11000 // 16+8? why not 16
  */
  auto rounder_sixteen = _mm_set1_epi16(16); // FIXED: this must be 16

  simd_blend_store4(ptrb + 0 * pitchb, mm4, weight_hi16, rounder_sixteen, zero);
  simd_blend_store4(ptrb + 1 * pitchb, mm5, weight_hi16, rounder_sixteen, zero);
  simd_blend_store4(ptrb + 2 * pitchb, mm6, weight_hi16, rounder_sixteen, zero);
  simd_blend_store4(ptrb + 3 * pitchb, mm7, weight_hi16, rounder_sixteen, zero);
}

AVS_FORCEINLINE void frcore_filter_overlap_b4r3_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_overlap_b4r2or3_simd<3>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// used in adaptive overlapping
// bottleneck in P = 1
AVS_FORCEINLINE void frcore_filter_overlap_b4r2_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{
  frcore_filter_overlap_b4r2or3_simd<2>(ptrr, pitchr, ptra, pitcha, ptrb, pitchb, thresh, inv_table, weight);
}

// mmA is input/output. In simd_blend_store4 mmA in input only
AVS_FORCEINLINE void simd_blend_diff4(uint8_t* esi, __m128i &mmA, __m128i mm2_multiplier, __m128i mm1_rounder, __m128i mm0_zero)
{
  auto mm3 = _mm_unpacklo_epi8(_mm_load_si32(esi), mm0_zero);
  // tmp= ((esi << 6) * multiplier) >> 16  ( == [esi]/1024 * multiplier)
  // mmA = (mmA + tmp + rounder_16) / 32
  // ((((mm1 << 2) * multiplier) >> 16 ) + 1) >> 1
  mm3 = _mm_slli_epi16(mm3, 6);
  mm3 = _mm_mulhi_epi16(mm3, mm2_multiplier); // pmulhw, signed
  mmA = _mm_adds_epu16(mmA, mm3);
  mmA = _mm_adds_epu16(mmA, mm1_rounder);
  mmA = _mm_srli_epi16(mmA, 5);
  mmA = _mm_packus_epi16(mmA, mm0_zero); // 4 words to 4 bytes
  *(uint32_t*)(esi) = _mm_cvtsi128_si32(mmA);
  mmA = _mm_sad_epu8(mmA, mm3); // this is the only difference from simd_blend_store4

  // but mm3 contains 4 words? and mmA contains 4 bytes? doesn't make sense. probably pack mm3 before psadbw
  // but it doesn't seem to affect the output
}

AVS_FORCEINLINE void frcore_filter_diff_b4r1_simd(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int thresh, const int* inv_table, int* weight)
{

  ptra += -1 * pitcha - 1; //  cpln(-1, -1)

  int weight_acc = 0;

  // reference pixels
  auto m0 = _mm_load_si32(ptrr); // 4 bytes
  auto m1 = _mm_load_si32(ptrr + pitchr * 1);
  auto m2 = _mm_load_si32(ptrr + pitchr * 2);
  auto m3 = _mm_load_si32(ptrr + pitchr * 3);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  // accumulators
  // each collects 4 words (weighted sums)
  // which will be finally scaled back and stored as 4 bytes
  auto mm4 = _mm_setzero_si128();
  auto mm5 = _mm_setzero_si128();
  auto mm6 = _mm_setzero_si128();
  auto mm7 = _mm_setzero_si128();

  // -1
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  ptra += pitcha; // next line

  // 0
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  ptra += pitcha; // next line

  // 0
  simd_check(ref01, ref23, 0, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 1, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);
  simd_check(ref01, ref23, 2, ptra, pitcha, weight_acc, thresh, mm4, mm5, mm6, mm7);

  // mm4 - mm7 has accumulated sum, weight is ready here

  // weight variable is a multi-purpose one, here we get a 32 bit value,
  // which is really two 16 bit words
  // lower 16 and upper 16 bit has separate meaning
  int prev_weight = *weight;

  // scale 4 - 7 by weight and store(here with blending)
  auto weight_recip = _mm_set1_epi32(inv_table[weight_acc] + (1 << 16));

  mm4 = _mm_unpacklo_epi16(mm4, _mm_set1_epi16(256));
  mm5 = _mm_unpacklo_epi16(mm5, _mm_set1_epi16(256));
  mm6 = _mm_unpacklo_epi16(mm6, _mm_set1_epi16(256));
  mm7 = _mm_unpacklo_epi16(mm7, _mm_set1_epi16(256));

  // We do this instead of pmulhw in order to avoid a loss of precision,
  // which would result in a green tint (lower pixel values).
  mm4 = _mm_madd_epi16(mm4, weight_recip);
  mm5 = _mm_madd_epi16(mm5, weight_recip);
  mm6 = _mm_madd_epi16(mm6, weight_recip);
  mm7 = _mm_madd_epi16(mm7, weight_recip);

  mm4 = _mm_srli_epi32(mm4, 9);
  mm5 = _mm_srli_epi32(mm5, 9);
  mm6 = _mm_srli_epi32(mm6, 9);
  mm7 = _mm_srli_epi32(mm7, 9);

  auto weight_lo16 = _mm_set1_epi32(prev_weight & 0xFFFF); // lower 16 bit

  mm4 = _mm_mulhi_epi16(mm4, weight_lo16);
  mm5 = _mm_mulhi_epi16(mm5, weight_lo16);
  mm6 = _mm_mulhi_epi16(mm6, weight_lo16);
  mm7 = _mm_mulhi_epi16(mm7, weight_lo16);

  // Experiments show that there is no need to add, then subtract the sign bit in this case.
  mm4 = _mm_packs_epi32(mm4, _mm_setzero_si128());
  mm5 = _mm_packs_epi32(mm5, _mm_setzero_si128());
  mm6 = _mm_packs_epi32(mm6, _mm_setzero_si128());
  mm7 = _mm_packs_epi32(mm7, _mm_setzero_si128());

  auto weight_hi16 = _mm_set1_epi16(prev_weight >> 16); // upper 16 bit

  auto zero = _mm_setzero_si128(); // packer zero mm0

  /*
    blend is >>5 inside then it would need rounder_16. and not 24 (16+8)
    // to be change to the good one after porting
    pcmpeqd mm1, mm1      1111111111111111
    psrlw	mm1, 14                       11
    psllw	mm1, 3                     11000 // 16+8? why not 16
  */
  auto rounder_sixteen = _mm_set1_epi16(16); // FIXED: this must be 16

  simd_blend_diff4(ptrb + 0 * pitchb, mm4, weight_hi16, rounder_sixteen, zero);
  simd_blend_diff4(ptrb + 1 * pitchb, mm5, weight_hi16, rounder_sixteen, zero);
  simd_blend_diff4(ptrb + 2 * pitchb, mm6, weight_hi16, rounder_sixteen, zero);
  simd_blend_diff4(ptrb + 3 * pitchb, mm7, weight_hi16, rounder_sixteen, zero);

  *weight = _mm_cvtsi128_si32(_mm_add_epi16(_mm_add_epi16(mm4, mm5), _mm_add_epi16(mm6, mm7)));
  // mm4, mm5, mm6, mm7 are changed, outputs are SAD
}

AVS_FORCEINLINE void frcore_dev_b4_simd(const uint8_t* ptra, int pitcha, int* dev)
{

  ptra += - 1; // cpln(-1, 0).ptr;

  // reference pixels
  auto m0 = _mm_load_si32(ptra + 1); // 4 bytes
  auto m1 = _mm_load_si32(ptra + pitcha * 1 + 1);
  auto m2 = _mm_load_si32(ptra + pitcha * 2 + 1);
  auto m3 = _mm_load_si32(ptra + pitcha * 3 + 1);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  ptra += pitcha;

  int sad1;
  simd_sad16(ref01, ref23, 0, ptra, pitcha, sad1);

  int sad2;
  simd_sad16(ref01, ref23, 2, ptra, pitcha, sad2);

  *dev = std::min(sad1, sad2);
}

AVS_FORCEINLINE void frcore_sad_b4_simd(const uint8_t* ptra, int pitcha, const uint8_t* ptrb, int pitchb, int* sad)
{
  // reference pixels
  auto m0 = _mm_load_si32(ptra); // 4 bytes
  auto m1 = _mm_load_si32(ptra + pitcha * 1);
  auto m2 = _mm_load_si32(ptra + pitcha * 2);
  auto m3 = _mm_load_si32(ptra + pitcha * 3);

  // 4x4 pixels to 2x8 bytes
  auto ref01 = _mm_or_si128(_mm_slli_epi64(m0, 32), m1); // mm0: 2x4 = 8 bytes
  auto ref23 = _mm_or_si128(_mm_slli_epi64(m2, 32), m3); // mm1: 2x4 = 8 bytes

  int sad1;
  simd_sad16(ref01, ref23, 0, ptrb, pitchb, sad1);

  *sad = sad1;
}

#else // not x86

#define frcore_dev_b4_simd                  frcore_dev_b4_scalar
#define frcore_sad_b4_simd                  frcore_sad_b4_scalar
#define frcore_filter_b4r0_simd             frcore_filter_b4r0_scalar
#define frcore_filter_overlap_b4r2_simd     frcore_filter_overlap_b4r2_scalar
#define frcore_filter_overlap_b4r3_simd     frcore_filter_overlap_b4r3_scalar
#define frcore_filter_adapt_b4r2_simd       frcore_filter_adapt_b4r2_scalar
#define frcore_filter_adapt_b4r3_simd       frcore_filter_adapt_b4r3_scalar
#define frcore_filter_b4r2_simd             frcore_filter_b4r2_scalar
#define frcore_filter_b4r3_simd             frcore_filter_b4r3_scalar
#define frcore_filter_diff_b4r1_simd        frcore_filter_diff_b4r1_scalar

#endif


AVS_FORCEINLINE int get_weight(int alpha)
{
  int a = ((alpha * (1 << 15)) / ((alpha + 1)));
  int b = ((1 << 15) / ((alpha + 1)));
  return (a << 16) | b;
}

AVS_FORCEINLINE int clipb(int weight) {
  return weight < 0 ? 0 : weight > 255 ? 255 : weight;
}

enum SIMD_or_scalar {
  Scalar = 0,
  SIMD = 1
};

template <bool simd>
static void process_plane(const uint8_t* srcp_orig, int src_pitch,
  const uint8_t* srcp_prev_orig, int src_prev_pitch,
  const uint8_t* srcp_next_orig, int src_next_pitch,
  uint8_t* dstp_orig, int dstp_pitch,
  bool mode_adaptive_overlapping, bool mode_temporal, bool mode_adaptive_radius,
  int dim_x, int dim_y,
  int R, int lambda, int P1_param, int tmax,
  const int* inv_table,
  uint8_t* wpln, int wp_stride) {
  constexpr int B = 4;
  constexpr int S = 4;

  for (int y = 0; y < dim_y + B - 1; y += S)
  {
    int sy = y;
    int by = y;
    if (sy < R) sy = R;
    if (sy > dim_y - R - B) sy = dim_y - R - B;
    if (by > dim_y - B) by = dim_y - B;

      uint8_t* dstp_curr_by = dstp_orig + dstp_pitch * by;
      uint8_t* dstp_curr_sy = dstp_orig + dstp_pitch * sy;
      const uint8_t* srcp_curr_sy = srcp_orig + src_pitch * sy; // cpln(sx, sy)
      const uint8_t* srcp_curr_by = srcp_orig + src_pitch * by; // cpln(bx, by)

    for (int x = 0; x < dim_x + B - 1; x += S)
    {
      int sx = x;
      int bx = x;
      if (sx < R) sx = R;
      if (sx > dim_x - R - B) sx = dim_x - R - B;
      if (bx > dim_x - B) bx = dim_x - B;

        uint8_t* dstp = dstp_curr_by + bx;
        uint8_t* dstp_s = dstp_curr_sy + sx;
        const uint8_t* srcp_s = srcp_curr_sy + sx; // cpln(sx, sy)
        const uint8_t* srcp_b = srcp_curr_by + bx; // cpln(bx, by)

      int dev, devp, devn;
      (simd ? frcore_dev_b4_simd
        : frcore_dev_b4_scalar)(srcp_s, src_pitch, &dev);

      // only for temporal use
      const uint8_t* srcp_next_s = nullptr;
      const uint8_t* srcp_prev_s = nullptr;

      if (mode_temporal)
      {
        srcp_prev_s = srcp_prev_orig + src_prev_pitch * sy + sx; // ppln(sx, sy)
        (simd ? frcore_sad_b4_simd
          : frcore_sad_b4_scalar)(srcp_s, src_pitch, srcp_prev_s, src_prev_pitch, &devp);

        srcp_next_s = srcp_next_orig + src_next_pitch * sy + sx; // npln(sx, sy)
        (simd ? frcore_sad_b4_simd
          : frcore_sad_b4_scalar)(srcp_s, src_pitch, srcp_next_s, src_next_pitch, &devn);

        dev = std::min(dev, devn);
        dev = std::min(dev, devp);
      }

      int thresh = ((dev * lambda) >> 10);
      thresh = (thresh > tmax) ? tmax : thresh;
      if (thresh < 1) thresh = 1;


      int weight;
      if (mode_temporal) {
        (simd ? frcore_filter_b4r0_simd
          : frcore_filter_b4r0_scalar)(srcp_b, src_pitch, srcp_b, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);

          int k = 1;
          if (devp < thresh)
          {
            weight = get_weight(k); // two 16 bit values inside
            (R == 2 ? (simd ? frcore_filter_overlap_b4r2_simd
                            : frcore_filter_overlap_b4r2_scalar)
                    : (simd ? frcore_filter_overlap_b4r3_simd
                            : frcore_filter_overlap_b4r3_scalar))(srcp_s, src_pitch, srcp_prev_s, src_prev_pitch, dstp_s, dstp_pitch, thresh, inv_table, &weight);
            k++;
          }

          if (devn < thresh)
          {
            weight = get_weight(k); // two 16 bit values inside
            (R == 2 ? (simd ? frcore_filter_overlap_b4r2_simd
                            : frcore_filter_overlap_b4r2_scalar)
                    : (simd ? frcore_filter_overlap_b4r3_simd
                            : frcore_filter_overlap_b4r3_scalar))(srcp_s, src_pitch, srcp_next_s, src_next_pitch, dstp_s, dstp_pitch, thresh, inv_table, &weight);
          }
        }
        else
        {
          // not temporal
          if (sx == x && sy == y && mode_adaptive_radius) {
            constexpr int thresh2 = 16 * 9; // First try with R=1 then if over threshold R=2 then R=3
            constexpr int thresh3 = 16 * 25; // only when R=3
            (R == 2 ? (simd ? frcore_filter_adapt_b4r2_simd
                            : frcore_filter_adapt_b4r2_scalar)
                    : (simd ? frcore_filter_adapt_b4r3_simd
                            : frcore_filter_adapt_b4r3_scalar))(srcp_b, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, thresh2, thresh3, inv_table, &weight);
          } else {
            // Nothing or adaptive_overlapping or some case of adaptive_radius
            (R == 2 ? (simd ? frcore_filter_b4r2_simd
                            : frcore_filter_b4r2_scalar)
                    : (simd ? frcore_filter_b4r3_simd
                            : frcore_filter_b4r3_scalar))(srcp_b, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
          }
        }

    }
  }

    if (mode_adaptive_overlapping)
    {
      for (int y = 2; y < dim_y - B; y += S)
      {
        constexpr int R_shadow = 1; // renamed from R to silence a shadow warning

        int sy = y;
        if (sy < R_shadow) sy = R_shadow;
        if (sy > dim_y - R_shadow - B) sy = dim_y - R_shadow - B;

      const uint8_t* srcp_curr_sy = srcp_orig + src_pitch * sy; // cpln(sx, sy)
      const uint8_t* srcp_curr_y = srcp_orig + src_pitch * y; // cpln(x, y)
      uint8_t* dstp_curr_y = dstp_orig + dstp_pitch * y;

        for (int x = 2; x < dim_x - B; x += S)
        {
          int sx = x;
          if (sx < R_shadow) sx = R_shadow;
          if (sx > dim_x - R_shadow - B) sx = dim_x - R_shadow - B;

        int dev = 10;
        const uint8_t* srcp_s = srcp_curr_sy + sx; // cpln(sx, sy)
        (simd ? frcore_dev_b4_simd
          : frcore_dev_b4_scalar)(srcp_s, src_pitch, &dev);

        int thresh = ((dev * lambda) >> 10);
        thresh = (thresh > tmax) ? tmax : thresh;
        if (thresh < 1) thresh = 1;

        const uint8_t* srcp_xy = srcp_curr_y + x; // cpln(x, y)
        uint8_t* dstp = dstp_curr_y + x;

        int weight = get_weight(1);
        (simd ? frcore_filter_diff_b4r1_simd
          : frcore_filter_diff_b4r1_scalar)(srcp_xy, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);

        wpln[wp_stride * (y / 4) + (x / 4)] = clipb(weight);
      }
    }

      for (int kk = 1; kk < 9; kk++)
      {
        constexpr int R_shadow = 2; // renamed from R to silence a shadow warning

      int k = kk;

        for (int y = (k / 3) + 1; y < dim_y - B; y += S)
        {
          int sy = y;
          if (sy < R_shadow) sy = R_shadow;
          if (sy > dim_y - R_shadow - B) sy = dim_y - R_shadow - B;

        const uint8_t* srcp_curr_sy = srcp_orig + src_pitch * sy;
        const uint8_t* srcp_curr_y = srcp_orig + src_pitch * y;
        uint8_t* dstp_curr_y = dstp_orig + dstp_pitch * y;

          for (int x = (k % 3) + 1; x < dim_x - B; x += S)
          {
            int sx = x;
            if (sx < R_shadow) sx = R_shadow;
            if (sx > dim_x - R_shadow - B) sx = dim_x - R_shadow - B;

          if (wpln[wp_stride * (y / 4) + (x / 4)] < P1_param)
            continue;

          int dev = 10;
          const uint8_t* srcp_s = srcp_curr_sy + sx; // cpln(sx, sy)
          (simd ? frcore_dev_b4_simd
            : frcore_dev_b4_scalar)(srcp_s, src_pitch, &dev);

          int thresh = ((dev * lambda) >> 10);
          thresh = (thresh > tmax) ? tmax : thresh;
          if (thresh < 1) thresh = 1;

          uint8_t* dstp = dstp_curr_y + x;
          const uint8_t* srcp_xy = srcp_curr_y + x; // cpln(x, y)
          int weight = get_weight(k); // two 16 bit words inside
          (simd ? frcore_filter_overlap_b4r2_simd
            : frcore_filter_overlap_b4r2_scalar)(srcp_xy, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
        }
      }

    }
  } // adaptive overlapping
}

PVideoFrame __stdcall AvsFilter::GetFrame(int n, IScriptEnvironment* env)
{
  const bool mode_adaptive_overlapping = P & 1;
  const bool mode_temporal = P & 2;
  const bool mode_adaptive_radius = P & 4;

  // fixme: why was it uncommented?
  //	if (n<1 || n>vi.num_frames-1) return child->GetFrame(n, env);

  PVideoFrame pf; // previous
  PVideoFrame nf; // next

  PVideoFrame cf = child->GetFrame(n, env);

  if (mode_temporal) {
    pf = child->GetFrame(n - 1, env);
    nf = child->GetFrame(n + 1, env);
  }
  PVideoFrame df = has_at_least_v8 ? env->NewVideoFrameP(vi, &cf) : env->NewVideoFrame(vi); // frame property support

  const int num_of_planes = std::min(vi.NumComponents(), 3);
  for (int pl = 0; pl < num_of_planes; pl++) { // PLANES LOOP
    const int plane = pl == 0 ? PLANAR_Y : (pl == 1 ? PLANAR_U : PLANAR_V);

    const bool chroma = pl > 0;

    if ((Thresh_luma == 0 && !chroma) || (Thresh_chroma == 0 && chroma)) {
      // plane copy
      env->BitBlt(df->GetWritePtr(plane), df->GetPitch(plane),
        cf->GetReadPtr(plane), cf->GetPitch(plane), cf->GetRowSize(plane), cf->GetHeight(plane));
      continue;
    }

    const int dim_x = cf->GetRowSize(plane);
    const int dim_y = cf->GetHeight(plane);;

    // prev/next: only for temporal
    const uint8_t* srcp_prev_orig = nullptr;
    const uint8_t* srcp_next_orig = nullptr;
    int src_prev_pitch;
    int src_next_pitch;

    if (mode_temporal) {
      srcp_prev_orig = pf->GetReadPtr(plane);
      src_prev_pitch = pf->GetPitch(plane);

      srcp_next_orig = nf->GetReadPtr(plane);
      src_next_pitch = nf->GetPitch(plane);
    }

    const uint8_t* srcp_orig = cf->GetReadPtr(plane);
    const int src_pitch = cf->GetPitch(plane);

    uint8_t* dstp_orig = df->GetWritePtr(plane);
    const int dstp_pitch = df->GetPitch(plane);

    int tmax = Thresh_luma;
    if (pl > 0) tmax = Thresh_chroma;

    // R_1stpass; // originally 3, experimental: 2

    (opt ? process_plane<SIMD>
      : process_plane<Scalar>)(srcp_orig, src_pitch,
        srcp_prev_orig, src_prev_pitch,
        srcp_next_orig, src_next_pitch,
        dstp_orig, dstp_pitch,
        mode_adaptive_overlapping, mode_temporal, mode_adaptive_radius,
        dim_x, dim_y,
        R_1stpass, lambda, P1_param, tmax,
        inv_table,
        wpln, wp_stride);

  } // PLANES LOOP

  return df;
}

AvsFilter::AvsFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip())
{
  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }


  if (!vi.IsPlanar() || !vi.IsYUV() || vi.BitsPerComponent() != 8)
    env->ThrowError("Frfun7: only 8 bit Y or YUV colorspaces are accepted.");

  lambda = (int)(args[1].AsFloat(1.1f) * 1024); // 10 bit integer arithmetic
  // parameter "T"
  Thresh_luma = (int)(args[2].AsFloat(6) * 16); // internal subsampling is 4x4, probably x16 covers that
  // parameter "Tuv"
  Thresh_chroma = (int)(args[3].AsFloat(2) * 16);
  // parameter "P"
  const int P_param = args[4].AsInt(0);

  opt = args[5].AsInt(1);
  if (lambda < 0)
    env->ThrowError("Frfun7: lambda cannot be negative");

  if (Thresh_luma < 0 || Thresh_chroma < 0)
    env->ThrowError("Frfun7: Threshold cannot be negative");

  P = P_param & 7;
  //P1_param = P == 1 ? P_param / 1000 : 0; // hidden parameter used only for adaptive overlapping
  P1_param = args[5].AsInt(0); // TP1 threshold parameter of P1
  
  R_1stpass = args[6].AsInt(3); // R1: radius of 1st pass. Originally fixed 3, can be set to R
  if(R_1stpass != 2 && R_1stpass != 3)
    env->ThrowError("Frfun7: R1 (1st pass radius) must be 2 or 3");

  // for adaptive overlapping: a number*1000 makes a weight threshold (?)
  //	P = 12*1000;
  //	P &= ~7;
  //	P |= 1;		// adaptive overlapping
  //	P |= 2;		// temporal
  //	P |= 4;		// adaptive radius

  wp_width = vi.width / 4; // internal subsampling is 4
  wp_height = vi.height / 4;
  const int ALIGN = 32;
  wp_stride = (((wp_width)+(ALIGN)-1) & (~((ALIGN)-1)));
  if (P & 1) // only used for adaptive
    wpln = new uint8_t[wp_stride * wp_height];
  else
    wpln = nullptr;

  // pre-build reciprocial table
  for (int i = 1; i < 1024; i++) {
    // 1/x table 1..1023 for 15 bit integer arithmetic
    inv_table[i] = (int)((1 << 15) / (double)i);
  }
  inv_table[1] = 32767; // 2^15 - 1
}

AvsFilter::~AvsFilter()
{
  delete[] wpln;
}

AVSValue __cdecl AvsFilter::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  return new AvsFilter(args, env);
}

// Declare and initialise server pointers static storage.
const AVS_Linkage* AVS_linkage = 0;

// DLL entry point called from LoadPlugin() to setup a user plugin.
extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors) {

  AVS_linkage = vectors;
  // this P parameter did not exist in rev6 but appeared in a build in 2013
  env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f[P]i[TP1]i[R1]i[opt]i", AvsFilter::Create, 0);
  //    env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f", AvsFilter::Create, 0);
  return "`x' xxx";
}
