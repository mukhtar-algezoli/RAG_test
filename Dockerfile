FROM python:3.10

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

WORKDIR /app

# Copy app code and set working directory
COPY . /app

# Run
ENTRYPOINT [“streamlit”, “run”, “chatbot.py”, “–server.port=8080”, “–server.address=0.0.0.0”]