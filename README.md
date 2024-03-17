# traversaal-hack
Traversaal Hackathon

https://huggingface.co/traversaal-ai-hackathon


Using python==3.10
```
python -m venv venv
pip install -r requirements.txt
```

- place ```.env``` file shared in this folder after cloning.
- ```load.ipynb``` contains code to upload dataset to db. No need to run it again as index is already created!
- Use ```query.ipynb``` file to query db and play along.

References

https://python.langchain.com/docs/integrations/vectorstores/qdrant

To run streamlit app
```
streamlit run app.py
```
