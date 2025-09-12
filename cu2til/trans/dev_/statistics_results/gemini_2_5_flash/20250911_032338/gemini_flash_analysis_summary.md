# Test Results Summary

## 📊 Model: gemini-2.5-flash

**Analysis Timestamp:** 20250911_032338  
**Total Cases:** 168 | **Success:** 138 (82.1%) | **Failed:** 30

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 48/168 | 28.6% |
| **Round ≤5 Success** | 138/168 | 82.1% |
| **Overall Success** | 138/168 | 82.1% |
| **Case Types Fully Passed** | 8/21 | 38.1% |
| **Avg Rounds for Success** | 2.25 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 48 | 48 | 28.6% |
| Round 2 | 46 | 94 | 56.0% |
| Round 3 | 19 | 113 | 67.3% |
| Round 4 | 12 | 125 | 74.4% |
| Round 5 | 13 | 138 | 82.1% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| ✅ | `add` | 100.0% | 8/8 | Excellent |
| ✅ | `bmm` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2d` | 100.0% | 8/8 | Excellent |
| ✅ | `depthwiseconv` | 100.0% | 8/8 | Excellent |
| ✅ | `gemm` | 100.0% | 8/8 | Excellent |
| ✅ | `relu` | 100.0% | 8/8 | Excellent |
| ✅ | `rmsnorm` | 100.0% | 8/8 | Excellent |
| ✅ | `sigmoid` | 100.0% | 8/8 | Excellent |
| 🟢 | `avgpool` | 87.5% | 7/8 | Good |
| 🟢 | `conv2dnchw` | 87.5% | 7/8 | Good |
| 🟢 | `gemv` | 87.5% | 7/8 | Good |
| 🟢 | `layernorm` | 87.5% | 7/8 | Good |
| 🟢 | `maxpool` | 87.5% | 7/8 | Good |
| 🟢 | `minpool` | 87.5% | 7/8 | Good |
| 🟢 | `softmax` | 87.5% | 7/8 | Good |
| 🟢 | `sumpool` | 87.5% | 7/8 | Good |
| 🟡 | `conv1d` | 75.0% | 6/8 | Moderate |
| 🟡 | `sign` | 75.0% | 6/8 | Moderate |
| ⚠️ | `gelu` | 37.5% | 3/8 | Poor |
| ⚠️ | `deformable` | 25.0% | 2/8 | Poor |
| ⚠️ | `mha` | 12.5% | 1/8 | Poor |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **bmm**: 8/8 cases
- **conv2d**: 8/8 cases
- **depthwiseconv**: 8/8 cases
- **gemm**: 8/8 cases
- **relu**: 8/8 cases
- **rmsnorm**: 8/8 cases
- **sigmoid**: 8/8 cases

### Areas for Improvement
- **gelu**: 3/8 cases (37.5%)
- **deformable**: 2/8 cases (25.0%)
- **mha**: 1/8 cases (12.5%)

### Round Analysis Insights
- **48** cases (28.6%) succeeded on first attempt
- **90** additional cases succeeded within 5 rounds
- **30** cases (17.9%) failed after maximum rounds

---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 8 | 38.1% |
| **Good Case Types** (80-99%) | 8 | 38.1% |
| **Moderate Case Types** (50-79%) | 2 | 9.5% |
| **Poor Case Types** (<50%) | 3 | 14.3% |

---

*Generated on 2025-09-12 at 15:42:44*
