// ============================================================================
// GradeAOVOpt — BlinkScript GPU kernel
// Grades an AOV pass to match Nuke's Grade node behaviour,
// and then puts it back into the beauty pass.
// ============================================================================


/********************************/
/*   Emanuele Comotti @2025     */
/********************************/

// MAJOR NOTES :
// No BBOX handling, only merge together with biggest, ROI not supported.

kernel GradeAOVOpt : ImageComputationKernel<ePixelWise> // Declare kernel, runs once per pixel
{
  // -----------------------------
  // IMAGE INPUTS / OUTPUTS
  // -----------------------------

  // Main beauty image input (premultiplied RGBA)
  Image<eRead,  eAccessPoint, eEdgeClamped> src;

  // The AOV we want to grade (premultiplied RGBA)
  Image<eRead,  eAccessPoint, eEdgeClamped> aov;

  // Optional mask input — we will use only its alpha channel
  Image<eRead,  eAccessPoint, eEdgeClamped> mask;

  // Output image
  Image<eWrite> dst;

  // -----------------------------
  // USER PARAMETERS (KNOBS)
  // -----------------------------
  param:

    // Blackpoint for grading (RGBA)
    float4 blackpoint;

    // Whitepoint for grading (RGBA)
    float4 whitepoint;

    // Lift (RGBA)
    float4 lift;

    // Gain (RGBA)
    float4 gain;

    // Multiply (RGBA)
    float4 multiply;

    // Offset (RGBA)
    float4 offset;

    // Gamma (RGBA)
    float4 gamma;

    // Clamp toggle for blacks
    bool black_clamp;

    // Clamp toggle for whites
    bool white_clamp;

    // View the graded AOV alone
    bool viewaov;

    // Reverse grading
    bool reverse;

    // Enable unpremultiply before grading
    bool unpremult;

    // Mix between original and graded result
    float mix;

    // Whether to apply mask alpha
    bool useMask;

  // -----------------------------
  // LOCAL (CACHED) VARIABLES
  // -----------------------------
  local:

    // Precomputed slope for linear stage
    float4 A;

    // Precomputed offset for linear stage
    float4 B;

    // Precomputed inverse gamma (1/gamma) for efficiency
    float4 invGamma;

  // -----------------------------
  // DEFINE DEFAULTS
  // -----------------------------
  void define()
  {
    // Set blackpoint default to 0
    defineParam(blackpoint, "blackpoint", float4(0.0f));

    // Set whitepoint default to 1
    defineParam(whitepoint, "whitepoint", float4(1.0f));

    // Lift default 0
    defineParam(lift, "lift", float4(0.0f));

    // Gain default 1
    defineParam(gain, "gain", float4(1.0f));

    // Multiply default 1
    defineParam(multiply, "multiply", float4(1.0f));

    // Offset default 0
    defineParam(offset, "offset", float4(0.0f));

    // Gamma default 1
    defineParam(gamma, "gamma", float4(1.0f));

    // Clamps off by default
    defineParam(black_clamp, "black clamp", false);
    defineParam(white_clamp, "white clamp", false);

    // View AOV off by default
    defineParam(viewaov, "view AOV", false);

    // Reverse grading off
    defineParam(reverse, "reverse", false);

    // Unpremult off
    defineParam(unpremult, "(un)premult", false);

    // Mix default 1
    defineParam(mix, "mix", float(1.0f));

    // Mask use off
    defineParam(useMask, "use mask", false);
  }

  // -----------------------------
  // INITIALISATION
  // -----------------------------
  void init()
  {
    // Compute slope for linear stage: multiply*(gain-lift)/(whitepoint-blackpoint)
    A = multiply * (gain - lift) / (whitepoint - blackpoint);

    // Compute offset for linear stage: offset + lift - A*blackpoint
    B = offset + lift - (A * blackpoint);

    // Compute 1/gamma for all channels
    invGamma = float4(1.0f / gamma.x,
                      1.0f / gamma.y,
                      1.0f / gamma.z,
                      1.0f / gamma.w);
  }

  // -----------------------------
  // FORWARD GAMMA FUNCTION
  // Applies gamma correction in forward mode (Nuke’s piecewise behaviour)
  // -----------------------------
  float3 forward_gamma(float3 x, float3 G, float3 invG)
  {
    // Output RGB after gamma
    float3 o;

    // Loop over R, G, B channels
    for (int i = 0; i < 3; i++)
    {
      // Current channel value
      float xi = x[i];

      // Gamma for current channel
      float Gi = G[i];

      // If gamma <= 0 → special case
      if (Gi <= 0.0f)
      {
        // Below 0 → black
        // Between 0 and 1 → unchanged
        // Above 1 → huge value (effectively “infinite” white)
        o[i] = (xi < 0.0f) ? 0.0f
             : ((xi > 1.0f) ? 1e30f : xi);
      }
      // If gamma not equal to 1
      else if (Gi != 1.0f)
      {
        // Inverse gamma
        float ig = invG[i];

        // If negative, leave unchanged
        if (xi < 0.0f)
        {
          o[i] = xi;
        }
        // Between 0 and 1 → pow curve
        else if (xi < 1.0f)
        {
          o[i] = pow(xi, ig);
        }
        // >= 1 → linear tail (1 stays 1, above 1 scales)
        else
        {
          o[i] = 1.0f + (xi - 1.0f) * ig;
        }
      }
      // If gamma is exactly 1 → no change
      else
      {
        o[i] = xi;
      }
    }

    // Return adjusted RGB
    return o;
  }

  // -----------------------------
  // REVERSE GAMMA FUNCTION
  // Inverse of forward_gamma
  // -----------------------------
  float3 reverse_gamma(float3 x, float3 G)
  {
    // Output RGB after reverse gamma
    float3 o;

    // Loop R, G, B channels
    for (int i = 0; i < 3; i++)
    {
      // Current channel value
      float xi = x[i];

      // Gamma value
      float Gi = G[i];

      // If gamma <= 0
      if (Gi <= 0.0f)
      {
        // Above 0 → white, else black
        o[i] = (xi > 0.0f) ? 1.0f : 0.0f;
      }
      // If gamma not equal to 1
      else if (Gi != 1.0f)
      {
        // If <= 0 → unchanged
        if (xi <= 0.0f)
        {
          o[i] = xi;
        }
        // Between 0 and 1 → pow curve with gamma
        else if (xi < 1.0f)
        {
          o[i] = pow(xi, Gi);
        }
        // >= 1 → linear tail
        else
        {
          o[i] = 1.0f + (xi - 1.0f) * Gi;
        }
      }
      // If gamma == 1 → no change
      else
      {
        o[i] = xi;
      }
    }

    // Return adjusted RGB
    return o;
  }

  // -----------------------------
  // PROCESS PER PIXEL
  // -----------------------------
  void process()
  {
    // Read beauty pixel
    float4 srcPx = src();

    // Read AOV pixel
    float4 aovPx = aov();

    // Get mask alpha (or 1.0 if no mask)
    float mAlpha = useMask ? mask().w : 1.0f;

    // Early-out if nothing will be applied
    if (mix <= 0.0f || mAlpha <= 0.0f)
    {
      // Output = unchanged AOV
      float4 outAov = aovPx;

      // If viewing AOV, just show it but keep bbox from src
      float4 result = viewaov
        ? (srcPx - srcPx + outAov)
        : (srcPx - aovPx + outAov);

      // Preserve alpha from src
      result.w = srcPx.w;

      // Write pixel to output
      dst() = result;

      // Stop here for this pixel
      return;
    }

    // Pack A, B, gamma values into RGB-only vectors
    float3 A3    = float3(A.x, A.y, A.z);
    float3 B3    = float3(B.x, B.y, B.z);
    float3 G3    = float3(gamma.x, gamma.y, gamma.z);
    float3 invG3 = float3(invGamma.x, invGamma.y, invGamma.z);

    // Hold premultiplied before/after grading values
    float4 original_pm;
    float4 graded_pm;

    // If unpremult is enabled
    if (unpremult)
    {
      // Calculate safe inverse alpha
      float invA = 1.0f / max(srcPx.w, 1e-8f);

      // Unpremult the AOV
      float4 linAov4 = aovPx * invA;

      // Get RGB channels from unpremult AOV
      float3 x = float3(linAov4.x, linAov4.y, linAov4.z);

      // Holder for graded RGB
      float3 y;

      // Forward grading
      if (!reverse)
      {
        // Apply linear stage
        float3 lin = A3 * x + B3;

        // Apply clamp if enabled
        if (white_clamp || black_clamp)
        {
          if (!white_clamp)
            lin = max(lin, float3(0.0f));
          else if (!black_clamp)
            lin = min(lin, float3(1.0f));
          else
            lin = clamp(lin, float3(0.0f), float3(1.0f));
        }

        // Apply forward gamma
        y = forward_gamma(lin, G3, invG3);
      }
      // Reverse grading
      else
      {
        // Reverse gamma
        float3 rev = reverse_gamma(x, G3);

        // Safe inverse A per channel
        float3 Ainv;
        for (int i = 0; i < 3; i++)
        {
          float Ai = A3[i];
          Ainv[i] = (fabs(Ai) > 1e-6f) ? (1.0f / Ai) : 1.0f;
        }

        // Reverse linear stage
        float3 Brev = -B3 * Ainv;
        rev = rev * Ainv + Brev;

        // Clamp if enabled
        if (white_clamp || black_clamp)
        {
          if (black_clamp)
            rev = max(rev, float3(0.0f));
          else if (white_clamp)
            rev = min(rev, float3(1.0f));
        }

        y = rev;
      }

      // Premult before grading
      original_pm = float4(x, linAov4.w) * srcPx.w;

      // Premult after grading
      graded_pm   = float4(y, linAov4.w) * srcPx.w;
    }
    // If not unpremult
    else
    {
      // RGB from premultiplied AOV
      float3 xpm = float3(aovPx.x, aovPx.y, aovPx.z);

      // Holder for graded RGB
      float3 ypm;

      // Forward grading
      if (!reverse)
      {
        // Linear stage
        float3 lin = A3 * xpm + B3;

        // Clamp if enabled
        if (white_clamp || black_clamp)
        {
          if (!white_clamp)
            lin = max(lin, float3(0.0f));
          else if (!black_clamp)
            lin = min(lin, float3(1.0f));
          else
            lin = clamp(lin, float3(0.0f), float3(1.0f));
        }

        // Forward gamma
        ypm = forward_gamma(lin, G3, invG3);
      }
      // Reverse grading
      else
      {
        // Reverse gamma
        float3 rev = reverse_gamma(xpm, G3);

        // Safe inverse A per channel
        float3 Ainv;
        for (int i = 0; i < 3; i++)
        {
          float Ai = A3[i];
          Ainv[i] = (fabs(Ai) > 1e-6f) ? (1.0f / Ai) : 1.0f;
        }

        // Reverse linear stage
        float3 Brev = -B3 * Ainv;
        rev = rev * Ainv + Brev;

        // Clamp if enabled
        if (white_clamp || black_clamp)
        {
          if (black_clamp)
            rev = max(rev, float3(0.0f));
          else if (white_clamp)
            rev = min(rev, float3(1.0f));
        }

        ypm = rev;
      }

      // Store before and after
      original_pm = aovPx;
      graded_pm   = float4(ypm, aovPx.w);
    }

    // Compute blend factor t from mask alpha and mix knob
    float t = min(1.0f, max(0.0f, mAlpha * mix));

    // If t is 1, take fully graded; else blend between original and graded
    float4 masked_pm = (t >= 1.0f) ? graded_pm
                                   : _fc_lerp(original_pm, graded_pm, t);

    // If viewaov, replace src with graded AOV but keep bbox from src
    // Else replace the old AOV in src with graded AOV
    float4 result = viewaov
      ? (srcPx - srcPx + masked_pm)
      : (srcPx - aovPx + masked_pm);

    // Keep alpha from src
    result.w = srcPx.w;

    // Write result to output
    dst() = result;
  }
}; 
