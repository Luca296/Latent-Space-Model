"""
Flask application for the Latent-Space-Model documentation site.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from config import Config
from rag.index import build_index
from rag.retrieve import retrieve
from rag.rerank import rerank
from rag.answer import generate_answer
from rag.store import index_exists


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.route('/')
    def index():
        """Render the main documentation page."""
        return render_template('index.html')

    @app.route('/api/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'index_exists': index_exists()
        })

    @app.route('/api/ask', methods=['POST'])
    def ask():
        """
        RAG pipeline endpoint.

        Request body: {question: string, history: array}
        Response: {answer: string, sources: array}
        """
        try:
            data = request.get_json()
            if not data or 'question' not in data:
                return jsonify({'error': 'Missing question field'}), 400

            question = data['question']
            history = data.get('history', [])

            # Step 1: Retrieve relevant chunks
            candidates = retrieve(question)

            # Step 2: Rerank for better relevance
            reranked = rerank(question, candidates)

            # Step 3: Generate answer
            result = generate_answer(question, reranked, history)

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/chat', methods=['POST'])
    def chat():
        """
        Streaming chat endpoint (basic implementation).

        Request body: {question: string, history: array}
        """
        # For now, just return the same as /api/ask
        # Streaming can be implemented with Flask-SSE or similar
        return ask()

    return app


# Create the application instance
app = create_app()


if __name__ == '__main__':
    # Check if index needs to be built
    if not index_exists():
        print("Index not found. Building...")
        print("This may take a few minutes...")
        build_index()

    # Run the Flask application
    print("\nStarting documentation server...")
    print(f"Visit http://localhost:5000 to view the documentation")
    print(f"API health check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.FLASK_DEBUG
    )
