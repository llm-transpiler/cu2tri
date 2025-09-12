# Test Results Summary

## 📊 Model: anthropic/claude-sonnet-4

**Analysis Timestamp:** 20250910_210335  
**Total Cases:** 168 | **Success:** 165 (98.2%) | **Failed:** 3

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 113/168 | 67.3% |
| **Round ≤5 Success** | 165/168 | 98.2% |
| **Overall Success** | 165/168 | 98.2% |
| **Case Types Fully Passed** | 18/21 | 85.7% |
| **Avg Rounds for Success** | 1.42 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 113 | 113 | 67.3% |
| Round 2 | 38 | 151 | 89.9% |
| Round 3 | 11 | 162 | 96.4% |
| Round 4 | 2 | 164 | 97.6% |
| Round 5 | 1 | 165 | 98.2% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| ✅ | `add` | 100.0% | 8/8 | Excellent |
| ✅ | `avgpool` | 100.0% | 8/8 | Excellent |
| ✅ | `bmm` | 100.0% | 8/8 | Excellent |
| ✅ | `conv1d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2d` | 100.0% | 8/8 | Excellent |
| ✅ | `conv2dnchw` | 100.0% | 8/8 | Excellent |
| ✅ | `deformable` | 100.0% | 8/8 | Excellent |
| ✅ | `gelu` | 100.0% | 8/8 | Excellent |
| ✅ | `gemm` | 100.0% | 8/8 | Excellent |
| ✅ | `gemv` | 100.0% | 8/8 | Excellent |
| ✅ | `maxpool` | 100.0% | 8/8 | Excellent |
| ✅ | `minpool` | 100.0% | 8/8 | Excellent |
| ✅ | `relu` | 100.0% | 8/8 | Excellent |
| ✅ | `rmsnorm` | 100.0% | 8/8 | Excellent |
| ✅ | `sigmoid` | 100.0% | 8/8 | Excellent |
| ✅ | `sign` | 100.0% | 8/8 | Excellent |
| ✅ | `softmax` | 100.0% | 8/8 | Excellent |
| ✅ | `sumpool` | 100.0% | 8/8 | Excellent |
| 🟢 | `depthwiseconv` | 87.5% | 7/8 | Good |
| 🟢 | `layernorm` | 87.5% | 7/8 | Good |
| 🟢 | `mha` | 87.5% | 7/8 | Good |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- **add**: 8/8 cases
- **avgpool**: 8/8 cases
- **bmm**: 8/8 cases
- **conv1d**: 8/8 cases
- **conv2d**: 8/8 cases
- **conv2dnchw**: 8/8 cases
- **deformable**: 8/8 cases
- **gelu**: 8/8 cases
- **gemm**: 8/8 cases
- **gemv**: 8/8 cases
- **maxpool**: 8/8 cases
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
- **113** cases (67.3%) succeeded on first attempt
- **52** additional cases succeeded within 5 rounds
- **3** cases (1.8%) failed after maximum rounds

---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 18 | 85.7% |
| **Good Case Types** (80-99%) | 3 | 14.3% |
| **Moderate Case Types** (50-79%) | 0 | 0.0% |
| **Poor Case Types** (<50%) | 0 | 0.0% |

---

*Generated on 2025-09-12 at 15:43:07*
