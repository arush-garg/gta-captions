# GTA Caption Generator

Using the `vipulmaheshwari/GTA-Image-Captioning-Dataset` from HuggingFace, I fine-tuned Microsoft's Generative Image2Text Transformer to generate captions for GTA scenes (at this rate, AI might generate GTA 6 before Rockstar Games does).

The model trained is relatively small, so it works on CPU, CUDA and MPS (Apple Metal). However, you might notice slower inference if running on CPU.

Now since the GIT model is very small (~200 million params), it does not always generate grammatically correct descriptions (as shown in the [demo](https://youtu.be/cVZJFDZnYcE)). However, it still proves that fine-tuning on less than a 1000 rows can still be effective for an army of small, specialized models.

## Set up

After you clone this repository, go through the the following steps before trying to run the app.

1. `pip install -r requirements.txt`
2. The `.env` file contains the download link for the model weights
3. Run the app using `streamlit run app.py`
<br><br>
<i>In case you're wondering, I used SSH into Google Colab to train the model using GPU. Check out this [tutorial](https://www.youtube.com/watch?v=wvDFNQNgqS8) for how to do it (though you shouldn't do it on a Free account)</i>