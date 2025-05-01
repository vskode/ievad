import { useEffect } from 'react';
import axios from 'axios';


export const MouseEvents = ({
  plotId,
  data,
  labels,
  dataIndex,
  setDataIndex,
  setSpecData,
  globalTimestamp,
  setGlobalTimestamp,
  hoveredPlotId,
  setHoveredPlotId
}) => {
    // Find the closest timestamp in this plot
    const find_closest_index = (timestamp) => {
    return data.continuous_ts.reduce((bestIdx, ts, idx) => 
        Math.abs(ts - timestamp) 
        < Math.abs(data.continuous_ts[bestIdx] - timestamp) 
        ? idx : bestIdx, 
        0);
    };

    const onMouseOverCircle = (plotId, hovered_index) => {
        const newTimestamp = data.continuous_ts[hovered_index];  
        const closestIndex = find_closest_index(newTimestamp);

        setHoveredPlotId(plotId);  // Track which plot is actively hovered
        setGlobalTimestamp(newTimestamp); // Sync timestamp for other plots
        setDataIndex(closestIndex); // Set index for all plots

        console.log("Hovered index:", globalTimestamp);
        };

        const onMouseLeavePlot = () => {
        setHoveredPlotId(null); // Allow normal syncing again
    };

    useEffect(() => {
        if (globalTimestamp === null) return;

        // Don't update the actively hovered plot to prevent flickering
        if (plotId !== undefined && plotId === hoveredPlotId) return;

        const closestIndex = find_closest_index(globalTimestamp);

        setDataIndex(closestIndex); // Set index for all plots
        
        console.log(`Plot ${plotId} updated to timestamp:`, globalTimestamp);
    }, [globalTimestamp, plotId]); // Ensure this re-runs when `globalTimestamp` changes

    
    const handleClick = (event, index) => {
        console.log("Clicked on circle:", index);

        const dataPoint = {
            'x': data.x[index],
            'y': data.y[index],
            'z': data.time_within_file[index],
            'source_file': data.audio_filenames[index],
            'meta': data.metadata,
            'index': index,
            'label': labels[index]
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
    return { onMouseOverCircle, onMouseLeavePlot, handleClick, find_closest_index };
}