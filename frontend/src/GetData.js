import { useEffect, useState } from 'react';
import axios from 'axios';

export const enrichDict = (dict) => {
  const lengths = dict.metadata['file_lengths (s)'];
  const dims = dict.metadata['embedding_dimensions'];
  const audiofiles = dict.metadata['audio_files'];
  const step_size = dict.metadata['segment_length (samples)'] / dict.metadata['sample_rate (Hz)'];

  let continuous_ts = [];
  let time_within_file = [];
  let audio_filenames = [];
  let last_ts_prev_file = 0;

    for (let file_idx = 0; file_idx < audiofiles.length; file_idx++) {
      for (let step = 0; step < lengths[file_idx]; step += step_size) {
        // fill arrays with data
        // for continuous_ts, add the last timestamp of the previous file
        // to the current step to get the continuous timestamp for each 
        // model, which is the same at the beginning of each audio file
        // for each model. after that depending on the step size the 
        // timestamp diverges.
        continuous_ts.push(step + last_ts_prev_file);
        audio_filenames.push(audiofiles[file_idx]);
        time_within_file.push(step);
        if (step + step_size > lengths[file_idx] &&
            dims[file_idx][0] !== step / step_size + 1) {
          console.log("Warning: step size does not match file length");
        }
      }
      last_ts_prev_file = lengths[file_idx] + last_ts_prev_file;
    }
    dict['continuous_ts'] = continuous_ts;
    dict['time_within_file'] = time_within_file;
    dict['audio_filenames'] = audio_filenames;

    return dict;
  }

export const useDataFetcher = (path, items, setItems) => {
    const [embeddings, setEmbeddings] = useState(null);
    // const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(true);

    // Fetch dictionaries from the backend
    const getDataFromBackend = async (getWhat, model_name=null) => {
        // const getDictionaries = async () => {
        let dicts = [];
        try {
        // const url = "http://127.0.0.1:8000/getEmbedPaths/";
        if (model_name !== null) {
            path['model_name'] = model_name;
        }
        const url = `http://127.0.0.1:8000/${getWhat}/`;
        // const url = `http://127.0.0.1:8000/getDictionaries/`;
        const response = await axios.post(url, path);
        console.log("Dictionaries received:", response.data.dicts); // Log response
        dicts = response.data.dict; // Update dictionaries state
        
        return dicts;
        } catch (error) {
        console.error("Error fetching dictionaries:", error);
        }
        return;
    };

  // Fetch embeddings only if dictionaries are populated and loaded
  useEffect(() => {
    const fetchEmbeddings = async () => {
      let dicts = [];
      let labels = [];
        try {
          let response_dict = {};
          let model_array = [];
          dicts = await getDataFromBackend("getEmbedPaths");
          // const labels = await axios.get('files/embeddings/colombia/labels.json')
          let model_names = Object.keys(dicts).slice(5, 7);
          for (let i = 0; i < model_names.length; i++) {
            labels = await getDataFromBackend("getLabels", model_names[i]);
            // console.log("Fetching file from:", dicts[i]); // Log file path being requested
            const embedding = await axios.get(dicts[model_names[i]]);
            let model_name = model_names[i];
            model_array.push(model_name);

            response_dict[model_names[i]] = {
              'data': embedding.data,
              'name': dicts[model_names[i]].split("___")[1].split('/')[0],
              'model_name': model_names[i]
            };
            response_dict[model_name]['data']['index'] = Array.from(
              {length: response_dict[model_name]['data'].x.length}, 
              (_, n) => n
            );  
            response_dict[model_name].data = enrichDict(response_dict[model_name].data);

            response_dict[model_name]['data'].labels = labels;
            

          }
          setEmbeddings(response_dict);
          // Directly set new items from response_dict
          const newItems = Object.values(response_dict).map((item) => ({
            id: item.model_name, // Use `name` as ID
            label: item.name, // Use `name` as label
            checked: ['birdnet', 'perch_bird'].includes(item.model_name) ? true : false // Default to checked if 'birdnet'
          }));
          setItems(newItems);
          console.log("Embeddings received:", response_dict); // Log response
          // }
        } catch (error) {
          console.error("Error fetching embeddings:", error);
        } finally {
          setLoading(false);
        }
    };
    fetchEmbeddings();
  }, []); 

  return { embeddings, items, loading };
}