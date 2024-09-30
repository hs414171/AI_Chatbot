# from rag_helper.methods import handle_db , retriever_grader , hallucinator , answer_grader , router , initiate_retriever , generator , web_search_tool
from rag_helper.graph_workflow import set_workflow




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
    from pprint import pprint
    inputs = {"question": "who is elvish yadav"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])



