# Thesis Documentation

**Note:** This is a work-in-progress repository for a Thesis I and II project. The final structure and content may evolve as the thesis progresses.

## Folder Structure

The directory is structured as follows:

- **Data Preprocessing:**
  - `scraper.ipynb`: Scraping the `.xlsx` file which contains the dataset [[1]](#1).
  - `preprocessor.ipynb`
- **Feature Extraction:**
  - `text_stream_BERT.ipynb`: Extracts text features using BERT.
  - `audio_stream_VIT.ipynb`: Extracts visual features using ViT.
  - `visual_stream_VIT.ipynb`: Extracts visual features using ViT.
- **File Extraction:**
  - `frameExtraction.ipynb`
  - `logmelExtraction.ipynb`
  - `textExtraction.ipynb`
- **Fusion and Classification:**
  - `smcaModelA.ipynb`: β → α as Query
  - `smcaModelB.ipynb`: β → α as Key and Value
  - `gmu-Arevalo.ipynb`: Gated Multimodal Unit [[2]](#2).
  - `simulParallel-Xie.ipynb` Simultaneous Parallel Approach [[3]](#3).
- **Modules:**
  - `cross_attention.py`
  - `dataloader.py`
  - `classifier.py`
  - `linear_transformation.py`
  - `output_max.py`
- **Evaluation and Validation:**
  - `trainer.py`
  - `evaluation.py`
  - `cross_validation.py`

## References

<a id="1">[1] </a>Shafaei, M., Smailis, C., Kakadiaris, I. A., & Solorio, T. (2021). \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A Case Study of Deep Learning Based Multi-Modal Methods for Predicting the Age-Suitability Rating of Movie Trailers. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_arXiv preprint arXiv:2101.11704._

<a id="2">[2] </a>Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, Fabio A. (2024). \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gated Multimodal Units for Information Fusion. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_ArXiv.org. https://arxiv.org/abs/1702.01992_

<a id="3">[3] </a>Xie, B., Sidulova, M., & Park, C. H. (2021). \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Robust multimodal emotion recognition from conversation with transformer-based crossmodality fusion. \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Sensors, 21(14), 4913._

## Contact

- Kyle Andre Castro
- Carl Mitzchel Padua
- Edjin Jerney Payumo
- Nathaniel David Samonte
