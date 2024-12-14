import streamlit as st

# Theming
st.set_page_config(page_title="My Portfolio", layout="wide")

st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header */
    .header {
        text-align: center;
        font-size: 50px;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
        background: linear-gradient(135deg,rgb(16, 12, 233),rgb(25, 78, 224));
        border-radius: 30px;
        margin-bottom: 40px;
        padding: 50px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
        animation: slideInFromTop 1s ease-out;
    }

    @keyframes slideInFromTop {
        0% {
            opacity: 0;
            transform: translateY(-100px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .header:hover {
        background: linear-gradient(135deg,rgb(11, 39, 199),rgb(2, 32, 166));
    }

    /* Container */
    .container {
        padding: 30px 60px;
        margin-top: 20px;
        max-width: 1200px;
        margin: 0 auto;
        background-color: #fff;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1.5s ease-out;
    }

    @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Section Titles */
    .section-title {
        font-size: 40px;
        color: #028CA6;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
        animation: fadeInUp 1.2s ease-out;
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Cards */
    .card {
        background: #ffffff;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid rgb(0, 0, 0);
        box-shadow: 4px 4px 0px rgba(0, 0, 0, 1);
        transition: transform 1s ease, box-shadow 1s ease;
        animation: fadeInUp 1s ease-out;
    }

    /* Card Hover Effects */
    .card:hover {
        border: 1px solid rgb(16, 12, 233);
        transform: scale(1.05) ;
        box-shadow: 4px 4px 0px rgb(16, 12, 233);

    /* Bounce Animation */
    .card-bounce:hover {
        animation: bounce 1s ease;
    }

    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    /* Flip Animation */
    .card-flip {
        perspective: 1000px;
    }

    .card-flip:hover .flip-card-inner {
        transform: rotateY(180deg);
    }

    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }

    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
    }

    .flip-card-front {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }

    .flip-card-back {
        background-color: #028CA6;
        color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transform: rotateY(180deg);
    }

    /* Zoom Animation */
    .card-zoom:hover {
        transform: scale(1.1);
        transition: transform 0.3s ease-in-out;
    }

    /* Scale Effect */
    .card-scale:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
    }

    /* Card Content */
    .title {
        font-size: 32px;
        font-weight: bold;
        color:rgb(11, 43, 224);
        margin-bottom: 10px;
    }

    .card-description {
        font-size: 20px;
        color: #555;
        line-height: 1.8;
        text-align: justify;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 18px;
        color:rgb(3, 3, 3);
        padding: 20px;
        margin-top: 50px;
        border-top: 3px solid rgb(16, 12, 233);
        animation: fadeIn 1.5s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Example of adding content with animations
st.markdown('<div class="header">Welcome to My Portfolio</div>', unsafe_allow_html=True)


# Sidebar Menu
menu = ["About Me", "Projects", "Blog Section","Coming Soon"]
icon = {
    "About Me": "👨‍💻",
    "Projects": "💻",
    "Blog Section": "✍️",
    "Coming Soon": "🧭",
}

st.sidebar.markdown("<div class='custom-sidebar-title'>Navigation</div>", unsafe_allow_html=True)
choice = st.sidebar.radio("Select a section:", menu, format_func=lambda x: f"{icon[x]} {x}")

if choice == "About Me":
        st.markdown(
        """
        <div class='card'>
            <h2 class='title'>About Me 👨‍💻</h2>
            <h3>Muhammad Haseeb</h3>
            <p class='description'>I am a motivated Machine Learning enthusiast passionate about solving real-world problems through data-driven solutions. With hands-on experience in building ML models, data preprocessing, and deploying interactive web apps using Streamlit, I transform complex algorithms into user-friendly tools.

Proficient in Python libraries like Scikit-learn, TensorFlow, and PyTorch, I excel in creating regression and classification models while visualizing insights using Matplotlib and Seaborn. I enjoy blending technical expertise with creativity to deliver impactful AI solutions.

Driven by curiosity and a growth mindset, I am eager to contribute to meaningful projects and embrace challenges that push my skills further. Let’s connect and collaborate on innovative ML applications!</p>
        </div>
        """, unsafe_allow_html=True
    )

        # Show Education details
        education_placeholder = st.empty()
        education_placeholder.markdown(
            """
            <div class="card">
                <div class="title">Education 🎓</div>
                <div class="description">Intermediate in Computer Science (ICS)</div>
            </div>
            """, unsafe_allow_html=True)
       

        # Show Skills details
        skills_placeholder = st.empty()
        skills_placeholder.markdown(
    """
    <div class="card">
        <div class="title">Skills 🛠️</div>
        <div class="description">
            <b>Programming Languages:</b> Python, JavaScript, C++, HTML, CSS, kotlin, java, javascript, C#, Dart, C <br>
            <b>App Development:</b> Dart, kotlin, java, Flutter, Android Studio <br>
            <b>Machine Learning & Data Science:</b> Machine Learning, Data Analysis, Data Visualization, Model Evaluation <br>
            <b>Frameworks & Libraries:</b> Scikit-Learn, Streamlit, Pandas, Numpy, OpenCV, Tkinter, Matplotlib, Seaborn, nltk, SpeechRecognition <br>
            <b>Web Development:</b> HTML, CSS, JavaScript, Flask, Divi <br>
            <b>Tools:</b> Git, Jupyter Notebooks, VSCode, PyCharm, Google Colaboratery <br>
        </div>
    </div>
    """, unsafe_allow_html=True)
        contact_placeholder = st.empty()
        contact_placeholder.markdown(
            """
            <div class="card">
                <div class="title">Contact Details 📬</div>
                <ul class="ul">
                    <li><b>Email:</b> muhammadhaseeb9323@gmail.com 📧</li>
                    <li><b>GitHub:</b> <a href="https://github.com/HaseebHub0" target="_blank">github.com/HaseebHub0</a> 🖥️</li>
                    <li><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/muhammad-haseeb-739884317/" target="_blank">linkedin.com/in/Muhammad-Haseeb</a> 🔗</li>
                    <li><b>Kaggle:</b> <a href="https://www.kaggle.com/haseebindata" target="_blank">Kaggle.com/in/haseebindata</a> 🔗</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
         # Wait for 1 second

    
# Projects Section
elif choice == "Projects":
    st.title("Projects 💻")

    # Define project categories
    categories = {
        
        "Machine Learning": [ 
            {"title": "House Price Prediction 🏠", "description": "Predict house prices using regression models.", "link": "https://github.com/HaseebHub0/House-Price-Prediction"},
            {"title": "Student Performance Prediction 👨‍🎓", "description": "Predict Student Performance using ML.", "link": "https://github.com/HaseebHub0/Student-Prediction"},
        ],
        "App Development": [
           
            {"title": "Task Manager 🗂️", "description": "A task management app for productivity.", "link": "https://github.com/username/task-manager"},
        ],
        "Python Projects": [
            {"title": "File Renamer 📂", "description": "A Python script to bulk rename files.", "link": "https://github.com/HaseebHub0/File-Renamer"},
            {"title": "Bank Programme 💸", "description": "A simple banking system that allows users to check balance, deposit, withdraw, and transfer money between accounts.", "link": "https://github.com/HaseebHub0/Bank-Programme"},
            {"title": "Hotel Menu 🏨", "description": "A digital menu system for a hotel that displays available dishes and allows users to select and order items.", "link": "https://github.com/HaseebHub0/Hotel-Menu"},
            {"title": "Virtual Pet 🐱", "description": "A virtual pet simulation where users can interact with a pet by feeding, playing, or letting it sleep to maintain its happiness, health, and energy.", "link": "https://github.com/HaseebHub0/Virtual-Pet"},
        ],
        "Computer Vision": [
            {"title": "Face Detection 🧑‍🤝‍🧑", "description": "Detect faces in images using OpenCV.", "link": "https://github.com/HaseebHub0/Face-Detection"},
            {"title": "Object Detection 🚗", "description": "Detect objects in images using YOLO.", "link": "https://github.com/HaseebHub0/Object-Detection"},
        ],
        
    }

    # Create tabs for each category
    tabs = st.tabs(list(categories.keys()))

    # Loop through each tab
    for idx, (category, projects) in enumerate(categories.items()):
        with tabs[idx]:
            st.subheader(f"{category} Projects")
            
            # Create a 3-column layout
            cols = st.columns(3)
            for i, project in enumerate(projects):
                col = cols[i % 3]  # Assign project to a column
                with col:
                    st.markdown(
    f"""
    <style>
    /* Card Styles */
    .card {{
        border: 1px solid rgb(0, 0, 0);
        padding: 30px;
        margin: 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff, #d9f2fa);
        box-shadow: 4px 4px 0px rgba(0, 0, 0, 1);
        transition: all 0.5s ease;
        text-align: center;
    }}
    .card:hover {{
        transform: translateY(-15px) scale(1.05);
        border: 1px solid rgb(16, 12, 233);
        background: linear-gradient(135deg, #e3f8ff, #cce7f3);
    }}
    
    /* Title Styles */
    .title {{
        font-size: 28px;
        font-weight: bold;
        color:rgb(18, 19, 19);
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }}
    
    /* Description Styles */
    .description {{
        margin-top: 15px;
        font-size: 18px;
        color: #555;
        text-align: center;
        font-style: italic;
        margin-bottom: 25px;
    }}
    
    /* Link Styles */
    .link {{
        display: inline-block;
        margin-top: 20px;
        padding: 12px 24px;
        text-decoration: none;
        color: white;
        background-color:rgb(255, 255, 255);
        border-radius: 15px;
        font-weight: bold;
        text-align: center;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }}
    .link:hover {{
        background-color: rgb(44, 44, 44);
        transform: scale(1.1);
    }}
    </style>
    
    <div class="card">
        <div class="title">{project['title']}</div>
        <div class="description">{project['description']}</div>
        <a href="{project['link']}" target="_blank" class="link">View Project 🔗</a>
    </div>
    """,
    unsafe_allow_html=True
)
elif choice == "Coming Soon":
    st.title("Coming Soon!")
    st.header("Projects 💻")

    coming_soon_projects = [
        {"title": "Weather App 🌦️", "description": "Coming Soon!"},
        {"title": "Web Scraping Tool 🕸️", "description": "A Python tool to scrape data from websites."},
        {"title": "Sales Forecasting 📊", "description": "Coming Soon!"},
        {"title": "Recommendation System 🎥", "description": "Coming Soon!"},
        {"title": "Stock Price Prediction 📉", "description": "Coming Soon!"},
        {"title": "Credit Card Fraud Detection 💳", "description": "Coming Soon!"},
    ]

    cols = st.columns(3)
    for i, project in enumerate(coming_soon_projects):
        if isinstance(project, dict):  # Ensure project is a dictionary
            col = cols[i % 3]  # Assign project to a column
        with col:
            st.markdown(
    f"""
    <style>
    /* Card Styles */
    .card {{
        border: 1px solid rgb(16, 12, 233);
        padding: 30px;
        margin: 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff, #d9f2fa);
        box-shadow: 4px 4px px rgba(0, 0, 0, 1);
        transition: all 0.5s ease;
        text-align: center;
    }}
    .card:hover {{
        transform: translateY(-15px) scale(1.05);
        box-shadow: 10px 10px 35px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #e3f8ff, #cce7f3);
    }}
    
    /* Title Styles */
    .title {{
        font-size: 28px;
        font-weight: bold;
        color: rgb(18, 19, 19);
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }}
    
    /* Description Styles */
    .description {{
        margin-top: 15px;
        font-size: 18px;
        color: #555;
        text-align: center;
        font-style: italic;
        margin-bottom: 25px;
    }}
    
    /* Link Styles */
    .link {{
        display: inline-block;
        margin-top: 20px;
        padding: 12px 24px;
        text-decoration: none;
        color: white;
        background-color:rgb(146, 145, 145);
        border-radius: 15px;
        font-weight: bold;
        text-align: center;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }}
    .link:hover {{
        background-color: rgb(44, 44, 44);
        transform: scale(1.1);
    }}
    </style>
    
    <div class="card">
        <div class="title">{project['title']}</div>
        <div class="description">{project['description']}</div>
        
    </div>
    """,
    unsafe_allow_html=True
)










elif choice == "Blog Section":
    st.title("Blog Section ✍️")
    
    # Create tabs for each category
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Machine Learning", "Deep Learning", "NLP", "App Development (Flutter)", "Computer Vision"])
    
    # Machine Learning blogs
    with tab1:
        ml_blogs = [  
    {  
        "title": "My ML Journey 🚀",  
        "content": """  
        I started my Machine Learning journey by taking free online courses and building small projects.  
        My first project was a linear regression model predicting house prices.  
        As I progressed, I learned about more complex algorithms like decision trees, neural networks, and support vector machines.  
        The journey has been both challenging and rewarding, and I’m excited to continue learning!  
        """  
    },  

    {  
        "title": "Deploying ML Models 🛠️",  
        "content": """  
        Deploying ML models allows users to interact with them in real-time.  
        My plan includes using platforms like Streamlit for web apps, Heroku for simple APIs, and AWS for scalable cloud deployment.  
        Deployment ensures the models are not limited to code but become accessible solutions for real-world problems.  
        """  
    },  

    {  
        "title": "Data Analysis Techniques 📊",  
        "content": """  
        Data analysis involves inspecting, cleaning, and visualizing data to extract useful insights.  
        Tools like pandas and NumPy help manipulate data, while Matplotlib and Seaborn visualize trends and patterns.  
        I applied these techniques in a sales forecasting project where I analyzed customer buying patterns and created insightful dashboards.  
        """  
    },  

    {  
        "title": "Data Preprocessing Techniques for ML Models 🔧",  
        "content": """  
        
        Data preprocessing is essential for preparing data before training ML models.  
        Key techniques include:
          
        - **Handling Missing Values:** Use mean, median, or advanced imputation methods.  
        - **Normalization & Scaling:** Ensure features are on the same scale for algorithms sensitive to feature magnitude.  
        - **Encoding Categorical Data:** Use label encoding or one-hot encoding.  
        - **Outlier Removal:** Use statistical techniques like Z-scores or IQR.  

        Proper data preprocessing improves model accuracy and robustness.  
        """  
    },  

    {  
        "title": "Introduction to Model Evaluation and Tuning 🔍",  
        "content": """  
        Model evaluation ensures that the ML model performs well on unseen data.  

        **Evaluation Metrics:**  
        - **Accuracy:** Correct predictions over total predictions (for balanced datasets).  
        - **Precision:** Relevant results among predicted positives (useful for spam detection).  
        - **Recall:** Correctly predicted positives among actual positives (useful in medical diagnoses).  
        - **F1 Score:** Harmonic mean of precision and recall for imbalanced datasets.  

        **Model Tuning Methods:**  
        - **Grid Search:** Tests all parameter combinations for best results.  
        - **Random Search:** Randomly selects parameter combinations for efficiency.  
        - **Bayesian Optimization:** Applies probabilistic models to optimize tuning.  
        
        Regular evaluation and tuning improve model performance significantly.  
        """  
    },  

    {  
        "title": "Building Your First ML Model 🧑‍💻",  
        "content": """  
        Creating your first ML model can be exciting!  
        Start by loading a dataset using pandas, exploring it visually, and splitting it into training and test sets.  
        Use scikit-learn to train a linear regression model and evaluate its accuracy.  
        Remember: Start small, practice often, and keep improving!  
        """  
    },  

    {  
        "title": "Supervised vs Unsupervised Learning 🤔",  
        "content": """  
        
        **Supervised Learning:**  
        - The model learns from labeled data (input-output pairs).  
        - Examples: Classification, Regression, and Forecasting tasks.  
        
        **Unsupervised Learning:**  
        - No labels are provided; the model finds patterns independently.  
        - Examples: Clustering, Dimensionality Reduction, and Anomaly Detection.  

        Both learning types power a wide range of ML applications, from recommendation engines to image recognition.  
        """  
    },  

    {  
        "title": "Understanding Overfitting & Underfitting 📈",  
        "content": """  
        
        **Overfitting:**  
        - The model performs well on training data but poorly on unseen data due to excessive complexity.  

        **Underfitting:**  
        - The model fails to capture patterns in the training data due to insufficient complexity.  

        **Solutions:**  
        - Use regularization techniques like L1 (Lasso) or L2 (Ridge).  
        - Perform cross-validation.  
        - Tune hyperparameters using grid or random search.  
        """  
    },  

    {  
        "title": "Exploring Feature Selection Methods 🔬",  
        "content": """  
        Feature selection involves selecting the most relevant features for a model.  

        **Methods Include:**  
        - **Filter Methods:** Statistical measures like correlation or Chi-square tests.  
        - **Wrapper Methods:** Recursive Feature Elimination (RFE).  
        - **Embedded Methods:** Algorithms that perform selection during training, like Lasso regression.  

        Selecting the right features simplifies models and boosts performance.  
        """  
    },  

    {  
        "title": "Introduction to Neural Networks 🧠",  
        "content": """  
        Neural networks are inspired by the human brain. They are composed of layers of interconnected nodes (neurons).  

        **Key Concepts:**  
        - **Input Layer:** Receives data.  
        - **Hidden Layers:** Extracts features using learned weights.  
        - **Output Layer:** Provides predictions.  

        Backpropagation is used to adjust weights to minimize prediction errors.  
        """  
    },  

    {  
        "title": "Working with Time Series Data 📆",  
        "content": """  
        Time series data involves observations over time, such as stock prices or weather forecasts.  

        **Popular Models:**  
        - **ARIMA:** Used for forecasting.  
        - **LSTM:** Advanced deep learning model for time-series tasks.  
        - **Prophet:** Facebook's tool for intuitive time-series analysis.  

        Forecasting relies on analyzing past trends and predicting future events.  
        """  
    }  
]  
 

        
        for blog in ml_blogs:
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">{blog['title']}</div>
                    <div class="description">{blog['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Deep Learning blogs
    with tab2:
        deep_learning_blogs = deep_learning_blogs = [  
    {  
        "title": "Introduction to Neural Networks 🧠",  
        "content": """  
        Neural networks are the foundation of deep learning, designed to mimic the way the human brain works.  

        **Basic Structure:**  
        - **Input Layer:** Receives input data.  
        - **Hidden Layers:** Processes data through interconnected nodes (neurons).  
        - **Output Layer:** Produces the final prediction.  

        **Learning Process:**  
        - Forward Propagation: Data moves through the network.  
        - Backpropagation: Weights are adjusted to minimize errors.  

        Applications range from image recognition to natural language processing.  
        """  
    },  

    {  
        "title": "Understanding Convolutional Neural Networks (CNNs) 🖼️",  
        "content": """  
        CNNs are specialized neural networks designed for image-related tasks.  

        **Key Components:**  
        - **Convolutional Layer:** Extracts features from images using filters.  
        - **Pooling Layer:** Reduces spatial dimensions to speed up processing.  
        - **Fully Connected Layer:** Combines features for prediction.  

        **Applications:**  
        - Image Classification (e.g., identifying objects in photos)  
        - Facial Recognition (used in security systems)  
        - Medical Imaging (detecting diseases from scans)  
        """  
    },  

    {  
        "title": "Exploring Recurrent Neural Networks (RNNs) 🔄",  
        "content": """  
        RNNs are designed for sequential data processing. Unlike regular neural networks, they retain information from previous steps using a memory mechanism.  

        **How They Work:**  
        - Data flows through the network sequentially.  
        - Each step’s output depends on previous computations.  

        **Applications:**  
        - **Time Series Analysis:** Predicting stock prices or weather.  
        - **Language Translation:** Translating text from one language to another.  
        - **Speech Recognition:** Converting spoken words into text.  

        Advanced models like LSTMs and GRUs solve RNN’s vanishing gradient problem.  
        """  
    },  

    {  
        "title": "Deep Learning Frameworks 📚",  
        "content": """  
        Deep learning frameworks simplify model creation, training, and deployment.  

        **Popular Frameworks:**  
        - **TensorFlow:** Developed by Google, supports large-scale ML tasks.  
        - **PyTorch:** Preferred for research, developed by Facebook.  
        - **Keras:** A high-level API that works on top of TensorFlow.  
        - **MXNet:** Backed by Amazon for scalable deep learning.  

        These frameworks provide pre-built libraries, tools, and tutorials to accelerate development.  
        """  
    },  

    {  
        "title": "Transfer Learning in Deep Learning 🔄",  
        "content": """  
        Transfer learning enables us to reuse pre-trained models for new tasks, reducing training time and boosting performance.  

        **How It Works:**  
        - A pre-trained model (like VGG16 or ResNet) is fine-tuned on a new dataset.  
        - We can freeze early layers to retain general features and train only specific layers.  

        **Applications:**  
        - **Image Classification:** Using models trained on ImageNet.  
        - **Natural Language Processing:** Fine-tuning models like BERT or GPT for text tasks.  
        - **Medical Diagnostics:** Applying models to detect diseases with limited datasets.  

        Transfer learning is a game changer in domains with scarce labeled data.  
        """  
    }  
]  

        
        for blog in deep_learning_blogs:
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">{blog['title']}</div>
                    <div class="description">{blog['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # NLP blogs
    with tab3:
        nlp_blogs = [  
    {  
        "title": "Introduction to NLP 🧠",  
        "content": """  
        Natural Language Processing (NLP) allows machines to understand and generate human language.  

        **Key Concepts:**  
        - **Syntax Analysis:** Understanding grammar and sentence structure.  
        - **Semantics:** Extracting meaning from text.  
        - **Pragmatics:** Understanding context beyond words.  

        **Applications:**  
        - Chatbots (like personal assistants)  
        - Translation services (Google Translate)  
        - Search engines (Google, Bing)  

        NLP bridges the gap between human communication and machine interpretation.  
        """  
    },  

    {  
        "title": "Text Preprocessing Techniques for NLP 📝",  
        "content": """  
        Text preprocessing is a critical step for improving NLP models' accuracy and efficiency.  

        **Core Techniques:**  
        - **Tokenization:** Breaking text into words or sentences.  
        - **Lemmatization:** Reducing words to their root forms (e.g., "running" → "run").  
        - **Stopword Removal:** Filtering out common words like "the," "is," "and."  
        - **Stemming:** Cutting words to their base forms (simpler than lemmatization).  
        - **Lowercasing:** Standardizing text to avoid case-sensitive issues.  

        These steps prepare raw text for analysis and model training.  
        """  
    },  

    {  
        "title": "Word Embeddings: A Deep Dive 🌐",  
        "content": """  
        Word embeddings are dense vector representations of words, capturing semantic meaning.  

        **Popular Techniques:**  
        - **Word2Vec:** Generates embeddings using context from large text corpora.  
        - **GloVe:** Combines word occurrence statistics for better embeddings.  
        - **FastText:** Handles subword information, improving results for rare words.  

        **Applications:**  
        - Sentiment Analysis (detecting emotions in text)  
        - Text Classification (categorizing emails or documents)  
        - Machine Translation (language-to-language conversion)  

        Word embeddings help models understand context and relationships between words.  
        """  
    },  

    {  
        "title": "Sentiment Analysis with NLP 💬",  
        "content": """  
        Sentiment analysis determines the emotional tone of text, commonly used in customer feedback or social media monitoring.  

        **How It Works:**  
        - **Data Collection:** Gather text from sources like reviews or tweets.  
        - **Preprocessing:** Clean and normalize the text.  
        - **Model Training:** Use models like Naive Bayes, SVM, or transformers (BERT).  
        - **Prediction:** Classify the text as positive, negative, or neutral.  

        **Use Cases:**  
        - Product Reviews (Amazon, Yelp)  
        - Social Media Monitoring (Twitter sentiment analysis)  
        - Customer Service Automation (analyzing support tickets)  

        Sentiment analysis helps businesses make data-driven decisions.  
        """  
    },  

    {  
        "title": "Named Entity Recognition (NER) 🔍",  
        "content": """  
        Named Entity Recognition (NER) extracts key information like names, dates, and locations from text.  

        **How It Works:**  
        - **Entity Types:**  
          - **Person:** Names of people (e.g., "Elon Musk")  
          - **Organization:** Company names (e.g., "Google")  
          - **Location:** Geographic places (e.g., "New York")  
          - **Date/Time:** Temporal data (e.g., "12th December 2024")  

        **Applications:**  
        - Resume Parsing (extracting names, skills, and experiences)  
        - Legal Document Analysis (highlighting contracts and terms)  
        - News Summarization (extracting main events and people)  

        NER enables information extraction from unstructured text for real-world applications.  
        """  
    }  
]  

        
        for blog in nlp_blogs:
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">{blog['title']}</div>
                    <div class="description">{blog['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # App Development (Flutter) blogs
    with tab4:
        flutter_blogs = [  
    {  
        "title": "Getting Started with Flutter 📱",  
        "content": """  
        Flutter is an open-source UI toolkit created by Google for building cross-platform applications from a single codebase.  

        **Why Choose Flutter?**  
        - **Cross-Platform:** Build for Android, iOS, Web, and Desktop.  
        - **Fast Development:** Hot Reload speeds up development.  
        - **Native Performance:** Applications run at native speed.  

        **Core Tools:**  
        - **Dart SDK:** Flutter uses the Dart programming language.  
        - **Flutter SDK:** Required for app development.  
        - **IDE Support:** Works well with Android Studio, VSCode, and IntelliJ IDEA.  

        Get started by installing the Flutter SDK and creating your first app!  
        """  
    },  

    {  
        "title": "Building Your First Flutter App 🎉",  
        "content": """  
        Let's build a simple "Hello World" app in Flutter.  

        **Steps:**  
        1. **Install Flutter SDK:** Follow the official guide.  
        2. **Create a New Project:** Run `flutter create hello_world`.  
        3. **Project Structure:** Familiarize yourself with `lib/main.dart`.  
        4. **Code Example:**  
           ```dart  
           import 'package:flutter/material.dart';  

           void main() => runApp(MyApp());  

           class MyApp extends StatelessWidget {  
             @override  
             Widget build(BuildContext context) {  
               return MaterialApp(  
                 home: Scaffold(  
                   appBar: AppBar(title: Text("Hello World")),  
                   body: Center(child: Text("Welcome to Flutter!")),  
                 ),  
               );  
             }  
           }  
           ```  

        **Run the App:** Use `flutter run` to see your app in action!  
        """  
    },  

    {  
        "title": "Flutter Widgets Explained 🛠️",  
        "content": """  
        Widgets are the building blocks of a Flutter application.  

        **Types of Widgets:**  
        - **Stateless Widgets:** Do not change during the app's lifetime. (e.g., `Text`, `Icon`)  
        - **Stateful Widgets:** Can change dynamically based on user interaction. (e.g., `TextField`, `Checkbox`)  

        **Common Widgets:**  
        - **Container:** Adds padding, margins, and color.  
        - **Row & Column:** Arrange elements horizontally or vertically.  
        - **ListView:** Displays a scrollable list.  
        - **Scaffold:** Provides a basic app layout structure.  

        Understanding widgets is essential for designing great UI in Flutter!  
        """  
    },  

    {  
        "title": "State Management in Flutter 🔄",  
        "content": """  
        State management controls how UI updates based on data changes.  

        **Popular State Management Approaches:**  
        - **Provider:** Simple and recommended by Google.  
        - **Riverpod:** A modern and more flexible alternative to Provider.  
        - **Bloc/Cubit:** Great for enterprise-level applications.  
        - **Redux:** Suitable for complex state management needs.  

        **Example (Provider):**  
        ```dart  
        import 'package:flutter/material.dart';  
        import 'package:provider/provider.dart';  

        void main() => runApp(MyApp());  

        class Counter with ChangeNotifier {  
          int _count = 0;  
          int get count => _count;  

          void increment() {  
            _count++;  
            notifyListeners();  
          }  
        }  

        class MyApp extends StatelessWidget {  
          @override  
          Widget build(BuildContext context) {  
            return ChangeNotifierProvider(  
              create: (_) => Counter(),  
              child: MaterialApp(home: CounterApp()),  
            );  
          }  
        }  

        class CounterApp extends StatelessWidget {  
          @override  
          Widget build(BuildContext context) {  
            final counter = Provider.of<Counter>(context);  

            return Scaffold(  
              appBar: AppBar(title: Text("Counter App")),  
              body: Center(child: Text("Count: ${counter.count}")),  
              floatingActionButton: FloatingActionButton(  
                onPressed: counter.increment,  
                child: Icon(Icons.add),  
              ),  
            );  
          }  
        }  
        ```  

        Choose the approach that fits your project’s complexity and scalability.  
        """  
    },  

    {  
        "title": "Flutter for Mobile vs Web Development 🌐",  
        "content": """  
        Flutter supports both mobile and web app development with a single codebase.  

        **Key Differences:**  

        | **Aspect**        | **Mobile Development**        | **Web Development**    |  
        |-------------------|-------------------------------|-------------------------|  
        | **Target Platform** | Android, iOS                  | Browsers (Chrome, Firefox) |  
        | **Rendering Engine** | Skia (native UI)             | CanvasKit or HTML       |  
        | **App Structure**   | Mobile-optimized layouts     | Responsive web layouts |  
        | **Navigation**      | Mobile navigation patterns   | Browser-based routing  |  
        | **Deployment**      | App Stores                   | Web Hosting            |  

        **Considerations:**  
        - Use responsive layouts (`MediaQuery`, `LayoutBuilder`) for web compatibility.  
        - Ensure that mobile features like push notifications are properly handled.  

        Flutter's flexibility makes it ideal for both platforms!  
        """  
    }  
]  

        
        for blog in flutter_blogs:
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">{blog['title']}</div>
                    <div class="description">{blog['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Computer Vision blogs
    with tab5:
        cv_blogs = [  
    {  
        "title": "Introduction to Computer Vision 👁️",  
        "content": """  
        Computer Vision (CV) is a field of Artificial Intelligence (AI) that enables machines to interpret and understand the visual world.  

        **Applications of CV:**  
        - **Image Classification:** Identifying objects in images.  
        - **Object Detection:** Locating objects in a scene.  
        - **Face Recognition:** Recognizing and verifying individuals.  
        - **Self-Driving Cars:** Understanding road scenes.  

        **Key Tools in CV:**  
        - **OpenCV:** An open-source library for computer vision tasks.  
        - **TensorFlow & PyTorch:** Popular deep learning frameworks for CV models.  

        Learn the basics of CV and explore real-world applications!  
        """  
    },  

    {  
        "title": "Image Preprocessing for Computer Vision 🖼️",  
        "content": """  
        Image preprocessing is essential in Computer Vision for improving model performance.  

        **Common Preprocessing Techniques:**  
        - **Resizing:** Adjust image dimensions for model input.  
        - **Normalization:** Scale pixel values between 0 and 1.  
        - **Data Augmentation:** Apply random transformations like flipping, rotation, and cropping.  
        - **Grayscale Conversion:** Convert color images to grayscale for simpler models.  

        **Example (OpenCV):**  
        ```python  
        import cv2  

        # Load image  
        img = cv2.imread('image.jpg')  

        # Resize image  
        resized_img = cv2.resize(img, (224, 224))  

        # Convert to grayscale  
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  

        # Show results  
        cv2.imshow("Processed Image", gray_img)  
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
        ```  

        These preprocessing steps can enhance model accuracy and reduce noise.  
        """  
    },  

    {  
        "title": "Object Detection with YOLO 🚀",  
        "content": """  
        
        **You Only Look Once (YOLO)** is a state-of-the-art, real-time object detection model.  

        **Why YOLO?**  
        - **Real-time Processing:** Can detect objects instantly.  
        - **Single Neural Network:** Processes the entire image at once.  
        - **High Accuracy:** Capable of detecting multiple objects simultaneously.  

        **How YOLO Works:**  
        - Divides an image into a grid.  
        - Predicts bounding boxes and confidence scores.  
        - Applies Non-Max Suppression (NMS) to eliminate duplicate boxes.  

        **Code Example:**  
        ```python  
        from ultralytics import YOLO  

        # Load the pre-trained YOLO model  
        model = YOLO("yolov8n.pt")  

        # Perform detection  
        results = model("test_image.jpg", show=True)  

        # Print results  
        print(results)  
        ```  

        YOLO is widely used in applications like surveillance, autonomous vehicles, and more!  
        """  
    },  

    {  
        "title": "Convolutional Neural Networks for Image Classification 🧠",  
        "content": """  
        Convolutional Neural Networks (CNNs) are the backbone of most Computer Vision applications.  

        **Why Use CNNs?**  
        - **Automatic Feature Extraction:** Detects patterns like edges, textures, and shapes.  
        - **High Accuracy:** Ideal for image classification tasks.  

        **Key Layers in CNNs:**  
        - **Convolutional Layer:** Extracts features from input images.  
        - **Pooling Layer:** Reduces dimensionality while retaining features.  
        - **Fully Connected Layer:** Classifies the image into categories.  

        **Example (Keras):**  
        ```python  
        from tensorflow.keras import models, layers  

        model = models.Sequential([  
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  
            layers.MaxPooling2D((2, 2)),  
            layers.Conv2D(64, (3, 3), activation='relu'),  
            layers.MaxPooling2D((2, 2)),  
            layers.Flatten(),  
            layers.Dense(128, activation='relu'),  
            layers.Dense(10, activation='softmax')  
        ])  

        model.summary()  
        ```  

        CNNs are widely used in tasks such as facial recognition, medical imaging, and more.  
        """  
    },  

    {  
        "title": "Face Recognition with OpenCV and Dlib 📸",  
        "content": """  
        Face recognition is one of the most popular Computer Vision tasks.  

        **Tools for Face Recognition:**  
        - **OpenCV:** Provides powerful face detection and recognition features.  
        - **Dlib:** Known for its robust facial landmark detection.  

        **Steps for Face Recognition:**  
        - Detect faces in an image.  
        - Extract facial features.  
        - Compare features for recognition.  

        **Example (OpenCV & Dlib):**  
        ```python  
        import cv2  
        import dlib  

        # Load face detector and predictor  
        detector = dlib.get_frontal_face_detector()  
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

        # Load image  
        img = cv2.imread("face.jpg")  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        faces = detector(gray)  
        for face in faces:  
            landmarks = predictor(gray, face)  

            # Draw facial landmarks  
            for n in range(0, 68):  
                x, y = landmarks.part(n).x, landmarks.part(n).y  
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  

        cv2.imshow("Face Recognition", img)  
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
        ```  

        Face recognition is used in authentication systems, security, and social media tagging.  
        """  
    }  
]  

        
        for blog in cv_blogs:
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">{blog['title']}</div>
                    <div class="description">{blog['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )






# Sidebar content
# Sidebar content
st.sidebar.info("Explore my Machine Learning Portfolio! 📂")

# Personal Note Section
st.sidebar.markdown("### Personal Note 📝")
st.sidebar.info("My best teacher is ChatGPT. 🌟")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit")
st.sidebar.markdown("© 2024 Muhammad Haseeb")




