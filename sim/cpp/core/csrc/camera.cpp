// Camera error models applied to perfect renders (port of StereoDepth / RgbCamera in
// sim/sensing/models.py). Pure functions over image buffers + an RNG stream.
#include <algorithm>
#include <limits>
#include <vector>

#include "core.hpp"

namespace rz {

static double focal_px(double fov_deg, int pixels) { return pixels / 2.0 / std::tan(fov_deg * DEG / 2); }

static void box_mean(const std::vector<double>& x, int h, int w, int k, std::vector<double>& out) {
  int p = k / 2, H = h + 2 * p, W = w + 2 * p;
  std::vector<double> c((H + 1) * (W + 1), 0.0);
  for (int r = 0; r < H; ++r) {
    int rr = std::min(std::max(r - p, 0), h - 1);
    double row = 0;
    for (int q = 0; q < W; ++q) {
      int cc = std::min(std::max(q - p, 0), w - 1);
      row += x[rr * w + cc];
      c[(r + 1) * (W + 1) + q + 1] = c[r * (W + 1) + q + 1] + row;
    }
  }
  out.resize(h * w);
  for (int r = 0; r < h; ++r)
    for (int q = 0; q < w; ++q) {
      int r0 = r, r1 = r + k, c0 = q, c1 = q + k;
      out[r * w + q] = (c[r1 * (W + 1) + c1] - c[r0 * (W + 1) + c1] - c[r1 * (W + 1) + c0] + c[r0 * (W + 1) + c0]) /
                       double(k * k);
    }
}

void depth_apply(const DepthSpec& s, const float* z, const float* gray, int h, int w, float* out, Rng& rng) {
  const double k = double(w) / s.width, f = focal_px(s.hfov, s.width) * k, fB = f * s.baseline;
  const double sig = s.sub_sigma * k, step = k / std::pow(2.0, s.sub_bits);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> disp(h * w);
  std::vector<uint8_t> bad(h * w, 0);
  for (int i = 0; i < h * w; ++i) {
    double zi = z[i], zc = zi > 1e-6 ? zi : inf;
    disp[i] = fB / zc;
    double dn = std::nearbyint((disp[i] + sig * rng.normal()) / step) * step;
    out[i] = dn > 0 ? float(fB / std::max(dn, 1e-9)) : 0.0f;
    bad[i] = zi < s.min_m || zi > s.max_m || !std::isfinite(zc) || dn <= 0;
  }
  if (s.lr_check) {
    for (int r = 0; r < h; ++r)
      for (int c = 1; c < w; ++c) {
        double j = disp[r * w + c - 1] - disp[r * w + c];
        if (!std::isfinite(j)) continue;
        int wd = int(std::min(std::ceil(j), 64.0));
        if (wd > 1)
          for (int q = c; q < std::min(c + wd, w); ++q) bad[r * w + q] = 1;
      }
  }
  if (gray && s.min_texture > 0) {
    int win = std::max(3, int(std::nearbyint(s.match_window * k)) | 1);
    std::vector<double> g(h * w), g2(h * w), m1, m2;
    for (int i = 0; i < h * w; ++i) { g[i] = gray[i] / 255.0; g2[i] = g[i] * g[i]; }
    box_mean(g, h, w, win, m1);
    box_mean(g2, h, w, win, m2);
    for (int i = 0; i < h * w; ++i)
      if (std::sqrt(std::max(m2[i] - m1[i] * m1[i], 0.0)) < s.min_texture) bad[i] = 1;
  }
  for (int i = 0; i < h * w; ++i)
    if (bad[i]) out[i] = 0.0f;
}

void rgb_apply(const RgbSpec& s, const uint8_t* img, int h, int w, const double omega[3], uint8_t* out,
               double& exposure, double& gain, Rng& rng) {
  const int n = h * w * 3;
  const double f = focal_px(s.hfov, w);
  std::vector<double> x(n);
  double sum = 0;
  for (int i = 0; i < n; ++i) { x[i] = img[i] / 255.0; sum += x[i]; }
  // auto exposure
  double e_ref = s.exp_ref / 1000, lum = std::max(sum / n, 1e-3), want = e_ref * s.ae_target / lum;
  exposure = clampd(want, s.exp_lo / 1000, s.exp_hi / 1000);
  gain = clampd(want / exposure, 1.0, s.max_gain);
  double sc = exposure / e_ref * gain;
  for (double& v : x) v *= sc;
  // motion blur from rotation during the exposure
  double vx = -f * omega[1], vy = f * omega[0];
  double L = std::hypot(vx, vy) * exposure;
  if (L >= 1.0) {
    int nk = int(std::min(std::ceil(L), 40.0));
    std::vector<double> acc(n, 0.0);
    for (int i = 0; i < nk; ++i) {
      double fr = double(i) / std::max(nk - 1, 1) - 0.5;
      int sy = int(std::nearbyint(fr * vy * exposure)), sx = int(std::nearbyint(fr * vx * exposure));
      for (int r = 0; r < h; ++r) {
        int rs = ((r - sy) % h + h) % h;
        for (int c = 0; c < w; ++c) {
          int cs = ((c - sx) % w + w) % w;
          for (int ch = 0; ch < 3; ++ch) acc[(r * w + c) * 3 + ch] += x[(rs * w + cs) * 3 + ch];
        }
      }
    }
    for (int i = 0; i < n; ++i) x[i] = acc[i] / nk;
  }
  // rolling shutter
  if (s.rolling) {
    std::vector<double> y(n);
    double tmid = (h / 2) * s.readout_ms / 1000 / h;
    for (int r = 0; r < h; ++r) {
      double tr = r * s.readout_ms / 1000 / h - tmid;
      int dx = int(std::nearbyint(vx * tr)), dy = int(std::nearbyint(vy * tr));
      int rs = std::min(std::max(r - dy, 0), h - 1);
      for (int c = 0; c < w; ++c) {
        int cs = ((c - dx) % w + w) % w;
        for (int ch = 0; ch < 3; ++ch) y[(r * w + c) * 3 + ch] = x[(rs * w + cs) * 3 + ch];
      }
    }
    x.swap(y);
  }
  for (int r = 0; r < h; ++r)
    for (int c = 0; c < w; ++c) {
      double vig = 1.0;
      if (s.vignetting > 0) {
        double r2 = ((c - w / 2.0) * (c - w / 2.0) + (r - h / 2.0) * (r - h / 2.0)) / (f * f);
        vig = 1 - s.vignetting * (1 - 1 / ((1 + r2) * (1 + r2)));
      }
      for (int ch = 0; ch < 3; ++ch) {
        int i = (r * w + c) * 3 + ch;
        double dn = clampd(x[i] * vig, 0.0, 1.0) * 255.0;
        double shot = std::sqrt(dn * s.shot * gain), rd = s.read * gain;
        dn += std::sqrt(shot * shot + rd * rd) * rng.normal();
        out[i] = uint8_t(clampd(std::nearbyint(dn), 0.0, 255.0));
      }
    }
}

}  // namespace rz
