things to implement:

- implement evaluation script to arbitarily check the performance of a modle on the test set of a given set of tensors. it should also output the resultant images.
- implement better checkpointing (checkPoint by Epoch), statistics tracking, and saving pipelines for model runs
- implement better methodology for training models (input as json file)
- update README with better instructions for local setup
- if model is not good at inferring, considering training on images with background replacement of comomon home office backgroudns ( from the POV of a webcam user)
- Recopy PalmPilot ( and maybe PalmPilot_Data ) to relieve root user permission from many files