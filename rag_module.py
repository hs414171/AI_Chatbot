from rag_helper.methods import handle_db
from rag_helper.graph_workflow import set_workflow
from pprint import pprint


def handle_rag(url: str = " ", question : str = " "):
    if url != " ":
        retriever = handle_db(url)

    workflow = set_workflow()
    app = workflow.compile()
    inputs = {"question": "What is an agent"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")

    return value["generation"]
    





if __name__ == "__main__":
    # url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    # retriever = handle_db(url)
    # question = "What is an agent"
    # decision,_ = retriever_grader(retriever,question)
    # print(decision)
    # hallucination_grader,_= hallucinator(retriever,question)
    # print(hallucination_grader)
    # answer_score,generation,_ = answer_grader(retriever,question)
    # print(answer_score)
    # print(generation)
    # question = "who is elvish yadav"
    # routed,_ = router(retriever,question)
    # print(routed)
    workflow = set_workflow()
    app = workflow.compile()

    # Test
    
    inputs = {"question": "What is an agent"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])



