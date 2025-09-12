# Test Results Summary

## 📊 Model: gemini-2.5-pro

**Analysis Timestamp:** 20250910_173830  
**Total Cases:** 168 | **Success:** 159 (94.6%) | **Failed:** 9

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 103/168 | 61.3% |
| **Round ≤5 Success** | 159/168 | 94.6% |
| **Overall Success** | 159/168 | 94.6% |
| **Case Types Fully Passed** | 15/21 | 71.4% |
| **Avg Rounds for Success** | 1.53 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 103 | 103 | 61.3% |
| Round 2 | 37 | 140 | 83.3% |
| Round 3 | 13 | 153 | 91.1% |
| Round 4 | 3 | 156 | 92.9% |
| Round 5 | 3 | 159 | 94.6% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| ✅ | `add` | 100.0% | 8/8 | Excellent |
| ✅ | `bmm` | 100.0% | 8/8 | Excellent |
| ✅ | `conv1d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2dnchw` | 100.0% | 8/8 | Excellent |
| ✅ | `depthwiseconv` | 100.0% | 8/8 | Excellent |
| ✅ | `gemm` | 100.0% | 8/8 | Excellent |
| ✅ | `gemv` | 100.0% | 8/8 | Excellent |
| ✅ | `minpool` | 100.0% | 8/8 | Excellent |
| ✅ | `relu` | 100.0% | 8/8 | Excellent |
| ✅ | `rmsnorm` | 100.0% | 8/8 | Excellent |
| ✅ | `sigmoid` | 100.0% | 8/8 | Excellent |
| ✅ | `sign` | 100.0% | 8/8 | Excellent |
| ✅ | `softmax` | 100.0% | 8/8 | Excellent |
| ✅ | `sumpool` | 100.0% | 8/8 | Excellent |
| 🟢 | `avgpool` | 87.5% | 7/8 | Good |
| 🟢 | `deformable` | 87.5% | 7/8 | Good |
| 🟢 | `gelu` | 87.5% | 7/8 | Good |
| 🟡 | `layernorm` | 75.0% | 6/8 | Moderate |
| 🟡 | `maxpool` | 75.0% | 6/8 | Moderate |
| 🟡 | `mha` | 75.0% | 6/8 | Moderate |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **bmm**: 8/8 cases
- **conv1d**: 8/8 cases
- **conv2d**: 8/8 cases
- **conv2dnchw**: 8/8 cases
- **depthwiseconv**: 8/8 cases
- **gemm**: 8/8 cases
- **gemv**: 8/8 cases
- **minpool**: 8/8 cases
- **relu**: 8/8 cases
- **rmsnorm**: 8/8 cases
- **sigmoid**: 8/8 cases
- **sign**: 8/8 cases
- **softmax**: 8/8 cases
- **sumpool**: 8/8 cases

### Areas for Improvement
- *All case types achieved ≥50% success rate*

### Round Analysis Insights
- **103** cases (61.3%) succeeded on first attempt
- **56** additional cases succeeded within 5 rounds
- **9** cases (5.4%) failed after maximum rounds

---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 15 | 71.4% |
| **Good Case Types** (80-99%) | 3 | 14.3% |
| **Moderate Case Types** (50-79%) | 3 | 14.3% |
| **Poor Case Types** (<50%) | 0 | 0.0% |

---

*Generated on 2025-09-12 at 15:42:23*
