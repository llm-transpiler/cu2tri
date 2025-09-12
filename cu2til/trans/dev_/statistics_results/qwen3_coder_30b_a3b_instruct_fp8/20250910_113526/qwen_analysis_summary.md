# Test Results Summary

## 📊 Model: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

**Analysis Timestamp:** 20250910_113526  
**Total Cases:** 168 | **Success:** 27 (16.1%) | **Failed:** 141

---

## 🎯 Key Performance Indicators

| Metric | Value | Percentage |
|--------|-------|------------|
| **Round 1 Success** | 19/168 | 11.3% |
| **Round ≤5 Success** | 27/168 | 16.1% |
| **Overall Success** | 27/168 | 16.1% |
| **Case Types Fully Passed** | 0/21 | 0.0% |
| **Avg Rounds for Success** | 1.41 | - |

---

## 📈 Success Distribution by Round

| Round | Cases | Cumulative | Success Rate |
|-------|-------|------------|--------------|
| Round 1 | 19 | 19 | 11.3% |
| Round 2 | 5 | 24 | 14.3% |
| Round 3 | 3 | 27 | 16.1% |

---

## 📋 Case Type Performance

| Status | Case Type | Success Rate | Results | Performance |
|--------|-----------|--------------|---------|-------------|
| 🟡 | `add` | 75.0% | 6/8 | Moderate |
| 🟡 | `relu` | 62.5% | 5/8 | Moderate |
| 🟡 | `sigmoid` | 62.5% | 5/8 | Moderate |
| 🟡 | `sign` | 62.5% | 5/8 | Moderate |
| ⚠️ | `conv1d` | 25.0% | 2/8 | Poor |
| ⚠️ | `sumpool` | 25.0% | 2/8 | Poor |
| ⚠️ | `depthwiseconv` | 12.5% | 1/8 | Poor |
| ⚠️ | `minpool` | 12.5% | 1/8 | Poor |
| ❌ | `avgpool` | 0.0% | 0/8 | Failed |
| ❌ | `bmm` | 0.0% | 0/8 | Failed |
| ❌ | `conv2d` | 0.0% | 0/8 | Failed |
| ❌ | `conv2dnchw` | 0.0% | 0/8 | Failed |
| ❌ | `deformable` | 0.0% | 0/8 | Failed |
| ❌ | `gelu` | 0.0% | 0/8 | Failed |
| ❌ | `gemm` | 0.0% | 0/8 | Failed |
| ❌ | `gemv` | 0.0% | 0/8 | Failed |
| ❌ | `layernorm` | 0.0% | 0/8 | Failed |
| ❌ | `maxpool` | 0.0% | 0/8 | Failed |
| ❌ | `mha` | 0.0% | 0/8 | Failed |
| ❌ | `rmsnorm` | 0.0% | 0/8 | Failed |
| ❌ | `softmax` | 0.0% | 0/8 | Failed |

---

## 🔍 Detailed Analysis

### Top Performers (100% Success Rate)
- *No case types achieved 100% success rate*

### Areas for Improvement
- **conv1d**: 2/8 cases (25.0%)
- **sumpool**: 2/8 cases (25.0%)
- **depthwiseconv**: 1/8 cases (12.5%)
- **minpool**: 1/8 cases (12.5%)
- **avgpool**: 0/8 cases (0.0%)
- **bmm**: 0/8 cases (0.0%)
- **conv2d**: 0/8 cases (0.0%)
- **conv2dnchw**: 0/8 cases (0.0%)
- **deformable**: 0/8 cases (0.0%)
- **gelu**: 0/8 cases (0.0%)
- **gemm**: 0/8 cases (0.0%)
- **gemv**: 0/8 cases (0.0%)
- **layernorm**: 0/8 cases (0.0%)
- **maxpool**: 0/8 cases (0.0%)
- **mha**: 0/8 cases (0.0%)
- **rmsnorm**: 0/8 cases (0.0%)
- **softmax**: 0/8 cases (0.0%)

### Round Analysis Insights
- **19** cases (11.3%) succeeded on first attempt
- **8** additional cases succeeded within 5 rounds
- **141** cases (83.9%) failed after maximum rounds

---

## 📊 Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent Case Types** (100%) | 0 | 0.0% |
| **Good Case Types** (80-99%) | 0 | 0.0% |
| **Moderate Case Types** (50-79%) | 4 | 19.0% |
| **Poor Case Types** (<50%) | 17 | 81.0% |

---

*Generated on 2025-09-12 at 15:42:02*
