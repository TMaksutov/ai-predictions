# AI Agents - Instructions and Updates

## Overview

This document provides instructions for AI agents working on this Time Series Forecasting Benchmark project, including how to check and update these instructions in the future.

## Current Project Status

### ✅ Completed Features
- **Layout Optimization**: Two-column layout with table on left, graph on right
- **Responsive Design**: All content fits on one page without scrolling
- **Compact Header**: Reduced header size for better space utilization
- **Interactive Visualization**: Real-time forecast plotting with dataset selection
- **Dual Model Comparison**: Linear Regression vs Prophet RMSE benchmarking

### 📋 Project Structure
```
/workspace/
├── streamlit_app.py          # Main application (two-column layout)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── agents.md                 # This file - agent instructions
├── TSForecasting_README.md   # Additional forecasting documentation
└── .streamlit/               # Streamlit configuration
```

## Agent Instructions

### Primary Tasks
1. **Maintain Layout Integrity**: Ensure all content remains on one page without scrolling
2. **Preserve Two-Column Structure**: Table (left) and graph (right) layout must be maintained
3. **Performance Optimization**: Keep benchmark computations efficient and cached
4. **User Experience**: Maintain responsive interactions and clear data visualization

### Code Guidelines
1. **Streamlit Layout**: Use `st.columns([1, 1.2])` for main layout structure
2. **Figure Size**: Keep matplotlib figures at `figsize=(8, 5)` for optimal fit
3. **Data Display**: Use `height=400` for dataframe to control vertical space
4. **Caching**: Maintain `@st.cache_data` decorators for performance
5. **Compact Design**: Use markdown headers (####) instead of st.subheader for space efficiency

### Dependency Management
- **Core Dependencies**: streamlit, pandas, numpy, matplotlib, scikit-learn, prophet
- **Version Ranges**: Use semantic versioning with appropriate constraints
- **Testing**: Ensure all dependencies work together in latest versions

## Instruction Check Protocol

### How to Verify Current Instructions
1. **Read this file**: `agents.md` contains the most current instructions
2. **Check git history**: `git log --oneline agents.md` to see instruction evolution
3. **Validate against code**: Ensure codebase matches documented patterns
4. **Test layout**: Verify one-page, two-column layout still works

### Validation Checklist
- [ ] Layout fits on one page without scrolling
- [ ] Table displays on left column
- [ ] Graph displays on right column  
- [ ] Header is compact (### level)
- [ ] All dependencies are up to date
- [ ] Performance remains responsive

## Updating Instructions

### When to Update
- **New Feature Requirements**: User requests additional functionality
- **Layout Changes**: Modifications to page structure or design
- **Dependency Updates**: Changes to required packages or versions
- **Performance Issues**: Optimization requirements or constraints
- **Bug Fixes**: Resolution of identified issues

### How to Update Instructions
1. **Edit this file**: Modify `agents.md` with new requirements
2. **Update status sections**: Add new completed features or requirements
3. **Revise guidelines**: Update code guidelines if patterns change
4. **Document changes**: Add entry to update history below
5. **Validate changes**: Ensure new instructions are clear and actionable

### Update Template
```markdown
### Update: [Date] - [Brief Description]
- **Changed**: [What was modified]
- **Reason**: [Why the change was needed]
- **Impact**: [How this affects future development]
- **Validation**: [How to verify the change works]
```

## Update History

### Update: 2024-12-28 - Initial Agent Instructions
- **Changed**: Created initial agents.md with instruction framework
- **Reason**: Need systematic way for agents to check and update instructions
- **Impact**: Provides structured approach for future AI agent interactions
- **Validation**: Document exists and contains all required sections

### Update: 2024-12-28 - Layout Optimization
- **Changed**: Implemented two-column layout, compact header, one-page design
- **Reason**: User requested table on left, graph on right, no scrolling
- **Impact**: All future layout changes must maintain this structure
- **Validation**: Run app and verify all content visible without scrolling

## Future Development Guidelines

### Recommended Enhancements
1. **Model Expansion**: Add more forecasting algorithms (ARIMA, LSTM, etc.)
2. **Dataset Management**: Allow custom dataset uploads
3. **Export Features**: Enable downloading of results and forecasts
4. **Real-time Data**: Integration with live data sources
5. **Advanced Metrics**: Additional evaluation metrics beyond RMSE

### Constraints to Maintain
1. **One-Page Layout**: Never allow content to require scrolling
2. **Two-Column Structure**: Preserve table/graph left/right arrangement
3. **Performance**: Keep initial load time under 5 seconds
4. **Compatibility**: Maintain compatibility with specified dependency versions
5. **Responsive Design**: Ensure layout works on different screen sizes

## Emergency Procedures

### If Layout Breaks
1. Revert to last working commit
2. Check for dependency conflicts
3. Verify column width ratios: `st.columns([1, 1.2])`
4. Ensure figure size is appropriate: `figsize=(8, 5)`

### If Dependencies Fail
1. Check requirements.txt for version conflicts
2. Test in clean virtual environment
3. Update version ranges if necessary
4. Document changes in this file

### If Performance Degrades
1. Verify caching decorators are present
2. Check for unnecessary computations in loops
3. Profile code to identify bottlenecks
4. Consider data preprocessing optimizations

---

**Last Updated**: 2024-12-28  
**Next Review**: As needed based on user requirements  
**Responsible**: AI Agent following these instructions