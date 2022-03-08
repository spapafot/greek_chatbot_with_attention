# Greek chatbot with attention mechanism

Greek chatbot trained on subtitles from greek shows (i used one show for consistency but you could mix many i guess)

The end goal is to create a chatbot that generates casual responses 

Requirements 

- numpy==1.19.5
- tensorflow==2.7.0
- pandas==1.3.5
- scikit-learn==1.0.1

Usage: 

> pip install -r requirements.txt

> git clone https://github.com/spapafot/greek_chatbot_with_attention.git my-project

> cd my-project

Add a folder with your raw data (greek subtitle files .srt format) to root and add the filepath to the ingest_data script

> python3 ingest_data.py -filepath

Settings in main.py

- BUFFER_SIZE = 32000 | tf.data.Dataset prefetch buffer
- embedding_dim = 300 | embedding_dim must be 300 because i am using Spacy's pretrained vectors
- units = 768 | Encoder - Decoder model units 
- BATCH_SIZE = 64 
- EPOCHS = 50

> python3 main.py

After 50 epochs i got decent results with the **beam translation**, the dataset i used had about 45k dialogue pairs
