from ievad.utils.embed2d import get_embeddings
from ievad.utils.plot import plotUMAP_Continuous_plotly

acc_embeddings, folders, file_list, lenghts = get_embeddings()
percentiles = 24
plotUMAP_Continuous_plotly(acc_embeddings, percentiles, 'plasma', 
                           file_list, lenghts)


# download checkpoint either mine or googles: 
# https://drive.google.com/file/d/1k1UpQFKSMkmdjYlm1GphjP-nW1uMHEiU/view?usp=sharing
# https://storage.googleapis.com/audioset/vggish_model.ckpt
