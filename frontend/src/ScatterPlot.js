import { useEffect, useMemo, useRef } from "react";
import axios from "axios";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 30, bottom: 50, left: 50 };

export const ScatterPlot = ({
  plotId,
  width,
  height,
  data,
  colorScale,
  setSpecData,
  globalTimestamp,
  setGlobalTimestamp,
  hoveredPlotId,
  setHoveredPlotId
}) => {
  const dataIndex = useRef(null);
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
  
  


  const onMouseOverCircle = (e, plotId) => {
    const circle = e.currentTarget;   
    const hoveredIndex = parseInt(circle.getAttribute("data-index"), 10);
    const hoveredTimestamp = data.timestamps[hoveredIndex];

    setHoveredPlotId(plotId);  // Track which plot is actively hovered
    setGlobalTimestamp(hoveredTimestamp); // Sync timestamp for other plots
    dataIndex.current = hoveredIndex; // Set index only for hovered plot
    console.log("Hovered index:", hoveredIndex);
};
const onMouseLeavePlot = () => {
  setHoveredPlotId(null); // Allow normal syncing again
};


useEffect(() => {
  if (globalTimestamp === null) return;

  // Don't update the actively hovered plot to prevent flickering
  if (plotId !== undefined && plotId === hoveredPlotId) return;

  // Find the closest timestamp in this plot
  const closestIndex = data.timestamps.reduce((bestIdx, ts, idx) => 
    Math.abs(ts - globalTimestamp) < Math.abs(data.timestamps[bestIdx] - globalTimestamp) ? idx : bestIdx, 
  0);

  dataIndex.current = closestIndex; // Force update for all plots
  console.log(`Plot ${plotId} updated to timestamp:`, globalTimestamp);
}, [globalTimestamp]); // Ensure this re-runs when `globalTimestamp` changes

  
const handleClick = (event, index) => {
  console.log("Clicked on circle:", index);

  // Force sync across plots
  const newTimestamp = data.timestamps[index];  
  setGlobalTimestamp(newTimestamp); // Update global timestamp immediately

  const dataPoint = {
    'x': data.x[index],
    'y': data.y[index],
    'z': data.time_within_file[index],
    'source_file': data.audio_filenames[index],
    'meta': data.metadata,
    'index': index,
    'label': data.label[index]
  };

  event.stopPropagation();  // Prevent event from being swallowed by other elements
  console.log("Circle clicked:", dataPoint);
  
  const url = "http://127.0.0.1:8000/";
  axios.post(url + 'getDataPoint/', dataPoint)
    .then(response => {
      console.log(response.data);
      setSpecData(response.data.spectrogram_data);
    })
    .catch(error => {
      console.log(error);
    });
};


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
          stroke={toColor(data.labels.ground_truth[i])} // Apply correct color
          fill={toColor(data.labels.ground_truth[i])}  // Fill with the same color
          fillOpacity={0.2}
          strokeWidth={1}
          pointerEvents="all" // Ensure the element can be clicked
          onMouseEnter={(e) => onMouseOverCircle(e, plotId)}
        />
      );
    }
    // console.log("current point:", dataIndex.current);
    return pts;
  }, [data, xScale, yScale]);
  const Cursor = ({ index, data }) => {
    const x = xScale(data.x[index]);
    const y = yScale(data.y[index]);
    const color = toColor(data.label[index]);
  
    const time_within_file = data.time_within_file[index].toFixed(2);
    const source_file = data.audio_filenames[index];
    const time_accum = data.timestamps[index].toFixed(2);
  
    return (
      <>
        {/* Cursor circle inside the plot */}
        <circle cx={x} cy={y} r={3} fill="black" onClick={(e) => handleClick(e, index)} />
        
        {/* Display text BELOW the plot */}
        <div style={{ textAlign: "left", marginTop: "10px", fontFamily: "Verdana", fontSize: "12px" }}>
          <p><strong>Model:</strong> {data.metadata.model_name}</p>
          <p><strong>Time within file:</strong> {time_within_file} sec</p>
          <p><strong>Source file:</strong> {source_file}</p>
          <p><strong>Time accumulated:</strong> {time_accum} sec</p>
        </div>
      </>
    );
  };
  
  return (
    <div style={{ textAlign: "center" }}> {/* Center the plot and text */}
      <svg width={width} height={height} onMouseLeave={onMouseLeavePlot}>
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {points}
          {dataIndex.current && <Cursor index={dataIndex.current} data={data} />}
        </g>
        <g ref={axesRef} transform={`translate(${MARGIN.left}, ${MARGIN.top})`} />
      </svg>
  
      {/* Text container BELOW the plot */}
      {dataIndex.current !== null && (
        <div style={{ marginTop: "10px", fontFamily: "Verdana", fontSize: "12px", textAlign: "center" }}>
          <p><strong>Model:</strong> {data.metadata.model_name}</p>
          <p><strong>Time within file:</strong> {data.time_within_file[dataIndex.current].toFixed(2)} sec</p>
          <p><strong>Source file:</strong> {data.audio_filenames[dataIndex.current]}</p>
          <p><strong>Time accumulated:</strong> {data.timestamps[dataIndex.current].toFixed(2)} sec</p>
        </div>
      )}
    </div>
  );
  
  
}
