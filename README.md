#Integrating Deep Learning Models With Symbolic Approaches to AI

⚠️ _Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and examples in this template before finalizing your README._

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |

|------------------|---------------|--------------------------------------------------------------------------|

| Era Kalaja    | @csera5  | Computer Vision object detection, Data Exploration, Integration   |

| Michelle Zuckerberg   | @X     | X  | Planner component, and instructions on how to use it for integration with the LLM component; Organizing team meetings

| X     | @X  | X                 |

| X      | @X       | X  |

| X       | @X    | X           |

---

## 🎯 **Project Highlights**

- Developed an integrated neural-symbolic AI system combining computer vision, large language models, and a STRIPS planner to address the challenge of translating natural-language questions and visual scenes into executable, interpretable action plans.

- Achieved reliable end-to-end system performance, including high-accuracy object detection, consistent symbolic fact generation, accurate goal translation, and valid action sequences, demonstrating the value of a multi-component pipeline for interpretable reasoning within MIT Lincoln Laboratory's research context.

- Generated actionable insights across the full pipeline, transforming raw images into symbolic propositions, natural language into formal logic, and planner outputs into human-readable explanations, enabling users and stakeholders to understand why and how the system reached its conclusions.

- Implemented a hybrid methodology integrating YOLOv8-based perception, LangChain-powered goal extraction, and Pyperplan STRIPS planning, satisfying industry expectations for explainable AI by ensuring each module produced traceable, verifiable intermediate outputs.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository

* How to install dependencies

* How to set up the environment

* How to access the dataset(s)

* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

This project was created as part of the Break Through Tech AI Program, which provides students with industry experience in building real-world AI systems. Through the program's AI Studio, our team partnered with MIT Lincoln Laboratory to explore neural-symbolic AI: the combination of neural network perception with symbolic reasoning.

Our project focused on building an end-to-end neural-symbolic reasoning system that can interpret an image of a scene, understand a natural-language question about it, and generate an interpretable sequence of actions to achieve a goal. To accomplish this, we developed:

- A computer vision module using YOLOv8 to detect objects and convert them into symbolic propositions

- An LLM component that translates user questions into formal logic goals and explains planner output

- A STRIPS planning module (Pyperplan) that computes valid action sequences using symbolic rules

- A custom frontend and integration layer that connects all components into an interactive user experience

This work addresses a broader real-world challenge: creating AI systems that are both capable and explainable. Neural-symbolic systems offer a path toward AI that can perceive complex environments, reason over them, and clearly communicate its decision process. The potential applications span robotics, autonomous agents, decision support, and any domain where transparent, verifiable reasoning is essential.

Our project demonstrates a practical example of how perception, language understanding, and symbolic planning can work together, moving toward more trustworthy, interpretable, and adaptable AI systems.

---

## 📊 **Data Exploration**

Dataset Description

Our project used two custom-curated datasets designed to support both perception and symbolic reasoning components of the system: (1) Realistic Image Dataset: Contained images representing monkeys, bananas, and boxes in a room-like environment. These images were manually annotated using Roboflow, producing YOLO-format bounding boxes. (2) Abstract Grid Dataset - A semantically equivalent dataset where objects were represented as colored cells on a grid. This allowed controlled experiments on scene understanding without visual noise or variation. Both datasets supported multi-class object detection, enabling us to identify the monkey, banana, and multiple box types required for downstream symbolic reasoning.

Format, Size & Structure

Annotation Format: YOLOv8 annotations (class, x_center, y_center, width, height)

Data Types: Raster images + bounding box labels

Classes: Monkey, Banana, Box A-E, Background

Purpose: Provide object-level spatial information to be converted into symbolic propositions for planning

Preprocessing 

Image preprocessing: resizing, normalization, and batch preparation

Train/val/test splits: ensuring generalizable performance across both datasets

Challenges & Assumptions

Small dataset size: required careful annotation and data augmentation to avoid overfitting.

Sample Dataset Image:

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)

* Feature selection and Hyperparameter tuning strategies

* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

Models Used:

- Our computer vision component is built on YOLOv8n, a lightweight, high-speed object detection model well-suited for real-time detection and small custom datasets. YOLOv8 served as both a feature extractor, learning spatial patterns through convolutional layers and a detection head, predicting bounding boxes, object classes, and confidence scores. This model was selected for its strong performance on small datasets and its ability to reliably detect multiple objects required by the symbolic planner (monkey, banana, boxes A–E). 

Features & Hyperparameter Strategy

- Since YOLOv8 extracts hierarchical spatial features automatically, manual feature selection was not required. Instead, the focus was on optimizing training behavior through: Learning rate tuning via YOLO's built-in scheduler, batch size adjustments to prevent overfitting on a small dataset, and data augmentation (horizontal flips, brightness/contrast shifts) to improve generalization. 

Training Setup:

- The Computer Vision model had a train/validation/test split of 70/20/10. It was evaluated on box_loss, precision, and recall. 

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)

* How your model performed

* Insights from evaluating model fairness

- Computer vision model metrics included: Recall, Precision, box_loss, mAP50. It performed very well across the dataset, with only some challenges distinguishing box E from box B. This challenge likely occurred since box B is seen in the training data over 80 times, while box E is only seen around 6 times so the model was biased towards box B. See below visualizations:

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?

* What would you do differently with more time/resources?

* What additional datasets or techniques would you explore?

---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**

This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

We would like to express our sincere gratitude to everyone who supported and guided us throughout this project.

Our Challenge Advisors at MIT Lincoln Laboratory

- Lee Martie, Technical Staff

- Sandra Hawkins, Assistant Staff

Our TA

- Mimi Lohanimit, EECS Graduate Researcher

Thank you for sharing your expertise, providing thoughtful technical direction, and helping us navigate the complexities of neural-symbolic AI.

Break Through Tech AI Program

Thank you for creating this opportunity and supporting our growth as emerging AI practitioners through hands-on, real-world experience.

AI Studio Teaching Assistants and Program Staff

Your feedback, mentorship, and encouragement were essential in helping us refine our ideas and successfully integrate each component of our system.

Finally, a big thank you to everyone behind the scenes who contributed resources, infrastructure, and continuous support throughout the development of this project.
