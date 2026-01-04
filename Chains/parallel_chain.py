from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()



LLM = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.1,
    max_new_tokens=1024
) # type: ignore

prompts = PromptTemplate(
    template='Generate small and simple notes from the following text below' \
    '{text}',
    input_variables=['text']
) # type: ignore

prompt1 = PromptTemplate(
    template='Generate five important  quiz type questions with answers on the following summary \n {text}',
    input_variables=['text']
)  # type: ignore

prompt2 = PromptTemplate(
    template='merge the provided notes and quiz into a single documents \n notes -> {notes} and \n quiz -> {quiz}',
    input_variables=['notes','quiz'] # type: ignore
)  # type: ignore

model1 = ChatHuggingFace(llm=LLM)
model2 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.8
    )
model3 = ChatHuggingFace(llm=LLM)


parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes' : prompts | model1 | parser,
    'quiz' : prompt1 | model2 | parser
})

merge_chain = prompt2 | model3 | parser

chain = parallel_chain | merge_chain

text = """
Support Vector Machine (SVM) is a powerful supervised machine learning algorithm primarily used for classification but also effective for regression and outlier detection, based on the core idea of finding an optimal decision boundary that maximizes the margin between different classes. SVM represents data points in an n-dimensional feature space and tries to identify a hyperplane that separates the classes with the largest possible margin, where the margin is defined as the distance between the hyperplane and the nearest data points from each class, known as support vectors, which ultimately determine the position and orientation of the decision boundary. For perfectly separable data, SVM uses a hard-margin approach that enforces strict separation, but since real-world data usually contains noise, overlaps, or outliers, the soft-margin SVM introduces slack variables to allow controlled misclassification while balancing margin width and classification accuracy through a regularization parameter C. For datasets that are not linearly separable in their original space, SVM employs the kernel trick, which maps the data into a higher-dimensional space using kernel functions such as linear, polynomial, RBF (Gaussian), or sigmoid, enabling SVM to form complex nonlinear boundaries without explicitly computing the high-dimensional transformation. The optimization problem behind SVM is convex and can be expressed in a dual form using Lagrange multipliers, ensuring that the solution is globally optimal and relies only on support vectors, making the model computationally efficient at prediction time. Although SVM provides strong generalization performance, especially in high-dimensional scenarios like text classification, image recognition, bioinformatics, and handwriting detection, it can be computationally expensive to train on very large datasets, sensitive to the choice of kernel and hyperparameters, and does not naturally provide probability outputs. Despite these limitations, SVM remains one of the most robust and theoretically grounded algorithms in machine learning, excelling particularly in situations where the number of features is large compared to the number of samples and where maximizing margin contributes to improved classification performance.

"""

model_response = chain.invoke({'text' : text})

print(model_response)



