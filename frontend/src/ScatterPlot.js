import { useEffect, useMemo, useRef, useState } from "react";
import { MouseEvents } from "./MouseEvents";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };

export const ScatterPlot = ({
  plotId,
  width,
  height,
  data,
  colorScale,
  labels,
  setSpecData,
  globalTimestamp,
  setGlobalTimestamp,
  hoveredPlotId,
  setHoveredPlotId
}) => {
  const [dataIndex, setDataIndex] = useState(null);
  const axesRef = useRef(null);
  const boundsWidth = width - MARGIN.right - MARGIN.left;
  const boundsHeight = height - MARGIN.top - MARGIN.bottom;
  
  const [yMin, yMax] = d3.extent(data.y);
  const yScale = useMemo(() => {
    return d3
    .scaleLinear()
    .domain([yMin, yMax || 0])
    .range([boundsHeight, 0]);
  }, [data, height]);
  
  // X axis
  const [xMin, xMax] = d3.extent(data.x);
  const xScale = useMemo(() => {
    return d3
    .scaleLinear()
    .domain([xMin, xMax || 0])
    .range([0, boundsWidth]);
  }, [data, width]);

  // Function to get color for a label
  function toColor(label) {
    return colorScale(label);
  }
  
  
  // Render the X and Y axis using d3.js, not react
  useEffect(() => {
    const svgElement = d3.select(axesRef.current);
    svgElement.selectAll("*").remove();
    
    // X Axis (Bottom)
    const xAxisBottom = d3.axisBottom(xScale).tickFormat(""); // No numbers
    svgElement.append("g")
    .attr("transform", `translate(0, ${boundsHeight})`)
    .call(xAxisBottom)
    .selectAll("text").remove();
    
    // X Axis (Top) - Just a Line
    svgElement.append("g")
    .attr("transform", `translate(0, 0)`) // Move to top
    .call(d3.axisTop(xScale).tickFormat("")) // No numbers
    .selectAll("text").remove();
    
    // Y Axis (Left)
    const yAxisLeft = d3.axisLeft(yScale).tickFormat("");
    svgElement.append("g")
    .call(yAxisLeft)
    .selectAll("text").remove();
    
    // Y Axis (Right) - Just a Line
    svgElement.append("g")
    .attr("transform", `translate(${boundsWidth}, 0)`) // Move to right
    .call(d3.axisRight(yScale).tickFormat("")) // No numbers
    .selectAll("text").remove();

      svgElement.selectAll(".tick line").style("display", "none"); // Hide tick marks

  }, [xScale, yScale, boundsHeight, boundsWidth]);


  const { onMouseOverCircle, onMouseLeavePlot, handleClick, find_closest_index } = MouseEvents({
    plotId,
    data,
    labels,
    dataIndex,
    setDataIndex,
    setSpecData,
    globalTimestamp,
    setGlobalTimestamp,
    hoveredPlotId,
    setHoveredPlotId,
    setDataIndex
  });

  const points = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= data.x.length; i++) {
      pts.push(
        <circle
          key={i}
          data-index={i}  // Use `i` directly
          r={2} // radius
          cx={xScale(data.x[i])} // position on the X axis
          cy={yScale(data.y[i])} // on the Y axis
          opacity={1}
          stroke={toColor(labels[i])} // Apply correct color
          fill={toColor(labels[i])}  // Fill with the same color
          fillOpacity={0.2}
          strokeWidth={1}
          pointerEvents="all" // Ensure the element can be clicked
          onMouseEnter={(e) => onMouseOverCircle(plotId, i)}
        />
      );
    }
    // console.log("current point:", dataIndex.current);
    return pts;
  }, [data, xScale, yScale, labels]);

  const Cursor = ({ data }) => {
    const closestIndex = find_closest_index(globalTimestamp);
    const x = xScale(data.x[closestIndex]);
    const y = yScale(data.y[closestIndex]);
  
    return (
      <>
        {/* Cursor circle inside the plot */}
        <circle 
        cx={x} 
        cy={y} 
        r={3} 
        fill="black" 
        onClick={(e) => handleClick(e, closestIndex)} />
      </>
    );
  };
  
  return (
    <div style={{ textAlign: "center" }}> {/* Center the plot and text */}
      <svg width={width} height={height} onMouseLeave={onMouseLeavePlot}>
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {points}
          {/* {dataIndex.current && <Cursor index={dataIndex.current} data={data} />} */}
          {dataIndex !== null && <Cursor data={data} />}
        </g>
        <g ref={axesRef} transform={`translate(${MARGIN.left}, ${MARGIN.top})`} />
      </svg>
  
      {/* Text container BELOW the plot */}
      {dataIndex !== null && hoveredPlotId !== null && (
        <div style={{ marginTop: "10px", fontFamily: "Verdana", fontSize: "12px", textAlign: "center" }}>
          <p><strong>Model:</strong> {data.metadata.model_name}</p>
          <p><strong>Time within file:</strong> {data.time_within_file[dataIndex].toFixed(2)} sec</p>
          <p><strong>Source file:</strong> {data.audio_filenames[dataIndex]}</p>
          <p><strong>Time accumulated:</strong> {data.continuous_ts[dataIndex].toFixed(2)} sec</p>
          <p><strong>globalTimestamp</strong> {globalTimestamp}</p>
          <p><strong>dataIndex</strong> {dataIndex}</p>
        </div>
      )}
    </div>
  );
  
  
}
