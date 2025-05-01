import { useState } from "react";
import { MakeSpectrogram } from "./Spectrogram";
import Container from "react-bootstrap/Container";
import * as d3 from "d3";
import { CheckboxDropdown } from "./CheckboxDropdown";
import { Dropdown } from "react-bootstrap";

import { ScatterPlot } from "./ScatterPlot";
import { useDataFetcher } from "./GetData";

export const MainLayout = ({ width = 700, height = 400 }) => {
  const [specData, setSpecData] = useState();
  const [hoveredPlotId, setHoveredPlotId] = useState(null);
  const [globalTimestamp, setGlobalTimestamp] = useState(null);
  
  // const [embeddings, setEmbeddings] = useState(null);
  const [items, setItems] = useState([
    { id: "model_name", label: "model-dataset-dimreduction", checked: true },
  ]);
  const [labelType, setLabelType] = useState("time_of_day");

  const path = { 
    "path": "files/embeddings/AnuranSet/dim_reduced_embeddings/" ,
    "model_name": "",
  };

  const { embeddings, loading } = useDataFetcher(path, items, setItems);

  if (loading) {  
    return <div>Loading...</div>;
  }  

  console.log("Global timestamp:", globalTimestamp);
  const uniqueLabels = [...new Set(embeddings['birdnet'].data.labels.ground_truth)];  // Extract unique labels
  const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(uniqueLabels);

  const plots = [];
  for (let i = 0; i < Object.keys(items).length; i++) {
    const model_name = items[i].id;
    if (!items[i].checked) {
      continue;
    }
    plots.push(
      <ScatterPlot
        plotId={i}
        width={width / 2}
        height={height}
        data={embeddings[model_name]['data']}
        colorScale={colorScale}
        labels={embeddings[model_name]['data'].labels[labelType]}
        setSpecData={setSpecData}
        globalTimestamp={globalTimestamp}
        setGlobalTimestamp={setGlobalTimestamp}
        hoveredPlotId={hoveredPlotId}
        setHoveredPlotId={setHoveredPlotId}
      />
    );
  }

  return (
    <Container fluid>
      {/* {BasicExample()} */}
      <CheckboxDropdown items={items} setItems={setItems} />
      <Dropdown>
        <Dropdown.Toggle variant="success" id="dropdown-basic">
          Select Label
        </Dropdown.Toggle>

        <Dropdown.Menu>
          {Object.keys(embeddings['birdnet']['data'].labels).map((label) => (
            <Dropdown.Item key={label} onClick={() => setLabelType(label)}>
              {label}
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown>
      <div style={{ display: "flex" }}>
        {plots}
        <MakeSpectrogram data={specData} />
      </div>
    </Container>
  );
};
