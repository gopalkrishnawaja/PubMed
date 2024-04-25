import json
import os
from sklearn.metrics import accuracy_score, f1_score
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.experimental.param_tuner import ParamTuner
from ray.tune.schedulers import ASHAScheduler
from llama_index.experimental.param_tuner import RayTuneParamTuner
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMIINI_API_KEY')

# Load th\e dataset
data_dir = 'data'
documents = SimpleDirectoryReader(data_dir).load_data()

def evaluate_pubmedqa(response):
    ground_truth = json.load(open('data/test_ground_truth.json'))
    predictions = response

    print("response............", response)

    assert set(list(ground_truth)) == set(
        list(predictions)), 'Please predict all and only the instances in the test set.'

    pmids = list(ground_truth)
    truth = [ground_truth[pmid] for pmid in pmids]
    preds = [predictions[pmid] for pmid in pmids]

    acc = accuracy_score(truth, preds)
    maf = f1_score(truth, preds, average='macro')

    print('Accuracy %f' % acc)
    print('Macro-F1 %f' % maf)


def objective_function(params):
    chunk_size = params["chunk_size"]
    top_k = params["top_k"]
    threshold = params["threshold"]
    max_chunk_overlap = 20

    print("chunk_size...............", chunk_size)

    Settings.llm = Gemini(model_name="models/gemini-pro", api_key=GEMINI_API_KEY)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 4096

    index = VectorStoreIndex.from_documents(documents=documents)

    index.storage_context.persist()

    query_engine = index.as_query_engine(
        temprature=0.7)

    response = query_engine.query("Is extended aortic replacement in acute type A dissection justifiable?")

    # Evaluate the performance on the PubmedQA dataset
    accuracy = evaluate_pubmedqa(response.response)
    return {"score": accuracy}


# Set up the hyperparameter search space
param_dict = {
    "chunk_size": [256, 512, 1024],
    "top_k": [1, 2, 5],
    "threshold": [0.5, 0.6, 0.7],
}

fixed_param_dict = {
    "docs": documents,
    "eval_qs": ["Is extended aortic replacement in acute type A dissection justifiable?"],
    "ref_response_strs": ["yes"]
}

# Run the ParamTuner
param_tuner = ParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True
)

results = param_tuner.tune()
best_result = results.best_run_result

print(f"Score: {best_result.score}")
print(f"Threshold: {best_result.params['threshold']}")
print(f"Top-k: {best_result.params['top_k']}")
print(f"Chunk size: {best_result.params['chunk_size']}")

# Use RayTuneParamTuner

ray_tuner = RayTuneParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    scheduler=ASHAScheduler()
)

analysis = ray_tuner.tune(num_samples=20)
best_trial = analysis.get_best_trial("score", "max", "last")

print(f"Best trial config: {best_trial.config}")
print(f"Best trial score: {best_trial.metric_analysis('score')['last']}")