import pytest
import os
from unittest import mock
import numpy as np
import pandas as pd
import geopandas as gpd # In case any db_utils mock needs it
import re # Import re for tokenization

# Module to test
from classify_erratic import ErraticClassifier, build_topic_model, main as classify_main

# --- Global Mocks & Fixtures ---

# Mock the NLTK data path check if necessary, though ideally handled by environment
# For unit tests, we assume NLTK data is available or the parts using it are mocked.

@pytest.fixture
def mock_spacy_nlp():
    """Mocks spacy.load() and the nlp object."""
    mock_nlp_object = mock.MagicMock()
    # Mock token properties needed by preprocess_text
    def mock_token_generator(text_input):
        tokens = []
        # Improved tokenization: separate words and punctuation
        for t_str in re.findall(r'\b\w+\b|[^\s\w]', text_input): # Changed regex slightly
            token = mock.MagicMock()
            token.text = t_str
            if re.match(r'^\w+$', t_str): # Word token
                token.lemma_ = t_str.lower() + "_lemma"
                token.is_punct = False
                token.is_stop = t_str.lower() in ['a', 'the', 'is'] # Explicitly mark 'is' as stop for clarity
            else: # Punctuation token
                token.lemma_ = t_str + "_lemma" # Punctuation lemma (less critical what it is, as it should be filtered)
                token.is_punct = True
                token.is_stop = False
            token.is_space = False # Assuming findall won't produce pure space tokens relevant here
            tokens.append(token)
        return tokens
    mock_nlp_object.side_effect = mock_token_generator
    with mock.patch('spacy.load', return_value=mock_nlp_object) as patched_spacy_load:
        yield patched_spacy_load, mock_nlp_object

@pytest.fixture
def mock_sentence_transformer():
    """Mocks SentenceTransformer."""
    mock_st_model = mock.MagicMock()
    mock_st_model.get_sentence_embedding_dimension.return_value = 384 # Matches project setting
    mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
    with mock.patch('classify_erratic.SentenceTransformer', return_value=mock_st_model) as patched_st:
        yield patched_st, mock_st_model

@pytest.fixture
def mock_topic_models():
    """Mocks BERTopic/LDA and related components."""
    mock_bertopic = mock.MagicMock()
    mock_bertopic.fit_transform.return_value = (np.array([0, 1, 0]), None) # topics, probs
    mock_bertopic.get_topic_info.return_value = pd.DataFrame({'Topic': [0, 1], 'Name': ['Topic_0', 'Topic_1']})
    mock_bertopic.get_topic.return_value = [('wordA', 0.5), ('wordB', 0.4)]
    mock_bertopic.transform.return_value = (np.array([0]), np.array([[0.8, 0.2]]))

    mock_lda = mock.MagicMock()
    mock_lda.fit_transform.return_value = np.random.rand(3, 2) # Dummy document-topic matrix
    mock_lda.transform.return_value = np.array([[0.7, 0.3]]) # topic distribution for one doc
    mock_lda.components_ = np.random.rand(2, 5) # topic-word matrix
    
    mock_vectorizer = mock.MagicMock()
    mock_vectorizer.fit_transform.return_value = mock.MagicMock() # Dummy sparse matrix
    mock_vectorizer.transform.return_value = mock.MagicMock()
    mock_vectorizer.get_feature_names_out.return_value = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']

    # Mock UMAP and HDBSCAN for BERTopic if ADVANCED_TOPIC_MODELING is True in the module
    mock_umap = mock.MagicMock()
    mock_hdbscan = mock.MagicMock()

    # Assume ADVANCED_TOPIC_MODELING can be controlled or test both paths
    # For now, let's assume we can patch it if needed, or test the LDA fallback.
    with mock.patch('classify_erratic.BERTopic', return_value=mock_bertopic) as p_bt, \
         mock.patch('classify_erratic.LatentDirichletAllocation', return_value=mock_lda) as p_lda, \
         mock.patch('classify_erratic.CountVectorizer', return_value=mock_vectorizer) as p_cv, \
         mock.patch('classify_erratic.UMAP', return_value=mock_umap) as p_umap, \
         mock.patch('classify_erratic.HDBSCAN', return_value=mock_hdbscan) as p_hdbscan:
        yield {
            'bertopic': p_bt, 'lda': p_lda, 'vectorizer': p_cv,
            'mock_bertopic_obj': mock_bertopic, 'mock_lda_obj': mock_lda, 'mock_vectorizer_obj': mock_vectorizer
        }

@pytest.fixture
def classifier_instance(mock_spacy_nlp, mock_sentence_transformer, tmp_path):
    # Ensure MODEL_SAVE_DIR in the module uses tmp_path for tests
    with mock.patch('classify_erratic.MODEL_SAVE_DIR', str(tmp_path / "test_models")):
        classifier = ErraticClassifier(model_dir=str(tmp_path / "test_models"))
        return classifier

# --- Test Class for ErraticClassifier ---
class TestErraticClassifier:
    def test_init(self, classifier_instance, mock_spacy_nlp, mock_sentence_transformer):
        patched_spacy_load, _ = mock_spacy_nlp
        patched_st, _ = mock_sentence_transformer
        
        patched_spacy_load.assert_called_once_with("en_core_web_md")
        patched_st.assert_called_once_with('all-MiniLM-L6-v2')
        assert classifier_instance.nlp is not None
        assert classifier_instance.embedding_model is not None

    def test_preprocess_text(self, classifier_instance, mock_spacy_nlp):
        _, mock_nlp_obj = mock_spacy_nlp
        text = "This is a Test! sentence for The preprocessing."
        # Expected output after filtering by len > 2, not stop, not punct:
        # "is" (len 2) is filtered.
        # "!" and "." are punctuation tokens and filtered.
        # "this_lemma test_lemma sentence_lemma for_lemma preprocessing_lemma"
        expected = "this_lemma test_lemma sentence_lemma for_lemma preprocessing_lemma"
        assert classifier_instance.preprocess_text(text) == expected
        assert classifier_instance.preprocess_text("") == ""
        assert classifier_instance.preprocess_text("  ! , . ") == ""

    def test_generate_embedding(self, classifier_instance, mock_sentence_transformer):
        _, mock_st_model = mock_sentence_transformer
        text = "Sample text for embedding."
        embedding = classifier_instance.generate_embedding(text)
        mock_st_model.encode.assert_called_once_with(text, normalize_embeddings=True)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

        empty_embedding = classifier_instance.generate_embedding("  ")
        assert np.all(empty_embedding == np.zeros(384, dtype=np.float32))

    # More complex tests for discover_topics and assign_topics would go here,
    # mocking the chosen topic modeling path (BERTopic or LDA)
    # These will depend on ADVANCED_TOPIC_MODELING flag in the module.

    @pytest.mark.parametrize("text, expected", [
        ("This rock has inscriptions.", True),
        ("A large engraved boulder.", True),
        ("Petroglyphs were found here.", True),
        ("A plain rock.", False),
        ("", False)
    ])
    def test_has_inscription_keywords(self, classifier_instance, text, expected):
        assert classifier_instance._has_inscription_keywords(text) == expected

    def test_classify(self, classifier_instance, mock_topic_models):
        # Assume topic model is loaded/mocked within classifier for this test
        classifier_instance.topic_model = mock_topic_models['mock_bertopic_obj'] # or lda
        classifier_instance.model_method = 'bertopic' # or lda
        # Mock topic_words to be populated as if a model was trained/loaded
        classifier_instance.topic_words = {0: ['wordA', 'wordB'], 1: ['wordC', 'wordD']}

        erratic_data = {
            'id': 100,
            'name': 'Classify Me',
            'description': 'Some descriptive text.',
            'cultural_significance': 'Culturally significant.',
            'historical_notes': 'Historically noted.'
        }
        result = classifier_instance.classify(erratic_data)
        assert result['erratic_id'] == 100
        assert 'vector_embedding' in result
        assert len(result['vector_embedding']) == 384
        assert result['classification']['has_inscriptions'] == False
        assert result['classification']['method'] == 'bertopic'
        assert result['topic_classification']['dominant_topic'] == 0 # From mock_bertopic.transform

    def test_classify_no_text(self, classifier_instance):
        erratic_data = {'id': 101, 'name': 'No Text Erratic'}
        result = classifier_instance.classify(erratic_data)
        assert result['classification']['method'] == 'no_text'
        assert np.all(np.array(result['vector_embedding']) == np.zeros(384))

    @mock.patch('classify_erratic.file_utils.json_to_file')
    @mock.patch('joblib.dump')
    @mock.patch('os.makedirs')
    def test_save_model_lda(self, mock_makedirs, mock_joblib_dump, mock_json_to_file, classifier_instance, mock_topic_models, tmp_path):
        # Setup for LDA save
        classifier_instance.model_method = 'lda'
        classifier_instance.topic_model = mock_topic_models['mock_lda_obj']
        classifier_instance.vectorizer = mock_topic_models['mock_vectorizer_obj']
        classifier_instance.topic_words = {0: ['test']}
        save_dir = str(tmp_path / "saved_lda_model")
        classifier_instance.model_dir = save_dir # Ensure it uses this path

        classifier_instance.save_model()

        mock_makedirs.assert_called_with(save_dir, exist_ok=True)
        assert mock_json_to_file.call_count == 2 # metadata and topic_words
        mock_joblib_dump.assert_any_call(classifier_instance.topic_model, os.path.join(save_dir, 'lda_model.joblib'))
        mock_joblib_dump.assert_any_call(classifier_instance.vectorizer, os.path.join(save_dir, 'vectorizer.joblib'))

    # Test for save_model BERTopic path would be similar, mocking BERTopic.save
    # Test for load_model for both LDA and BERTopic paths also needed

# Test for top-level build_topic_model function
@mock.patch('utils.db_utils.load_all_erratics_gdf')
def test_build_topic_model(mock_load_gdf, classifier_instance, mock_topic_models):
    # Mock GDF data returned by db_utils
    sample_erratics_data = {
        'id': [1, 2],
        'name': ['Erratic 1', 'Erratic 2'],
        'description': ['desc1', 'desc2'],
        'cultural_significance': ['cult1', 'cult2'],
        'historical_notes': ['hist1', 'hist2']
    }
    mock_load_gdf.return_value = gpd.GeoDataFrame(pd.DataFrame(sample_erratics_data))
    
    # Mock the classifier's discover_topics method directly as it's complex
    classifier_instance.discover_topics = mock.MagicMock(return_value={"num_topics": 2, "method": "mocked"})
    
    result = build_topic_model(classifier_instance)
    
    mock_load_gdf.assert_called_once()
    classifier_instance.discover_topics.assert_called_once()
    # Check that texts passed to discover_topics are correct (e.g. by inspecting args)
    args_passed = classifier_instance.discover_topics.call_args[0][0] # First positional arg (text_corpus)
    assert len(args_passed) == 2
    assert "desc1 cult1 hist1" in args_passed
    assert result == {"num_topics": 2, "method": "mocked"}

# --- Test Class for main CLI ---
@mock.patch('classify_erratic.ErraticClassifier')
@mock.patch('classify_erratic.build_topic_model')
@mock.patch('utils.db_utils.load_erratic_details_by_id')
@mock.patch('utils.db_utils.update_erratic_analysis_results')
@mock.patch('classify_erratic.file_utils.json_to_file')
class TestClassifyMain:
    def test_main_build_topics(self, mock_json_file, mock_update_db, mock_load_details, mock_build_topics, MockErraticClassifier, tmp_path):
        mock_args = mock.MagicMock()
        mock_args.build_topics = True
        mock_args.erratic_id = None
        mock_args.output = str(tmp_path / "topics_output.json")
        mock_args.update_db = False
        mock_args.verbose = False
        mock_args.model_dir = None

        mock_classifier_instance = mock.MagicMock()
        MockErraticClassifier.return_value = mock_classifier_instance
        mock_build_topics.return_value = {"num_topics": 3, "method": "bertopic"}

        # Mock the parser instance that will be used inside main()
        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with mock.patch('argparse.ArgumentParser', return_value=mock_parser) as MockArgumentParserClass:
            with mock.patch('sys.exit') as mock_sys_exit:
                classify_main()
                # Check if sys.exit was called because main() calls it in this specific path
                mock_sys_exit.assert_any_call(0) # build_topics path with no erratic_id exits with 0
        
        MockErraticClassifier.assert_called_once()
        mock_build_topics.assert_called_once_with(mock_classifier_instance)
        mock_classifier_instance.save_model.assert_called_once()
        mock_json_file.assert_called_once_with(mock_build_topics.return_value, str(tmp_path / "topics_output_topics.json"))

    def test_main_classify_erratic_with_update(self, mock_json_file, mock_update_db, mock_load_details, mock_build_topics, MockErraticClassifier, tmp_path):
        mock_args = mock.MagicMock()
        mock_args.build_topics = False
        mock_args.erratic_id = 1
        mock_args.output = str(tmp_path / "classify_output.json")
        mock_args.update_db = True
        mock_args.verbose = False
        mock_args.model_dir = str(tmp_path / "custom_model_dir")

        mock_classifier_instance = mock.MagicMock()
        MockErraticClassifier.return_value = mock_classifier_instance
        
        mock_erratic_data = {'id': 1, 'name': 'Test'}
        mock_load_details.return_value = mock_erratic_data
        
        classification_result = {
            "classification": {"top_categories": [0], "cultural_significance_score": 5, "has_inscriptions": False},
            "vector_embedding": [0.1]*384
        }
        mock_classifier_instance.classify.return_value = classification_result
        mock_update_db.return_value = True

        # Mock the parser instance
        mock_parser = mock.MagicMock()
        mock_parser.parse_args.return_value = mock_args

        with mock.patch('argparse.ArgumentParser', return_value=mock_parser) as MockArgumentParserClass:
            # sys.exit is not called directly by main() on this successful path, 
            # so we don't mock or assert it here.
            classify_main()

        MockErraticClassifier.assert_called_once_with(model_dir=str(tmp_path / "custom_model_dir"))
        mock_classifier_instance.load_model.assert_called_once()
        mock_load_details.assert_called_once_with(1)
        mock_classifier_instance.classify.assert_called_once_with(mock_erratic_data)
        mock_update_db.assert_called_once()
        mock_json_file.assert_called_once()

    def test_main_erratic_id_required(self, mock_json_file, mock_update_db, mock_load_details, mock_build_topics, MockErraticClassifier):
        mock_args = mock.MagicMock()
        mock_args.build_topics = False
        mock_args.erratic_id = None # Missing erratic_id when not building topics
        # ... other args ...

        # Mock the ArgumentParser instance and its error method
        mock_parser_instance = mock.MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        # We expect parser.error to be called, which then calls sys.exit
        mock_parser_instance.error = mock.MagicMock(side_effect=SystemExit(2)) 

        with mock.patch('argparse.ArgumentParser', return_value=mock_parser_instance) as MockArgumentParserClass:
            with pytest.raises(SystemExit) as e:
                classify_main()
            assert e.value.code == 2 # argparse.error calls sys.exit(2)
        
        mock_parser_instance.error.assert_called_once_with("erratic_id is required unless --build-topics.") 