import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import json
import io
import base64
import logging

# Configure page
st.set_page_config(
    page_title="AI Skill Gap Analyzer - Milestone 3",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design from second document
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(20px);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        transition: transform 0.2s;
    }
    
    .skill-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
    }
    
    .strong-match {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        color: white;
        padding: 10px 16px;
        border-radius: 10px;
        margin: 5px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    
    .partial-match {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 10px 16px;
        border-radius: 10px;
        margin: 5px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
    }
    
    .missing-skill {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 10px 16px;
        border-radius: 10px;
        margin: 5px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .priority-high {
        color: #ff6b6b;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(255, 107, 107, 0.3);
    }
    
    .priority-medium {
        color: #ffa502;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(255, 165, 2, 0.3);
    }
    
    .priority-low {
        color: #38ef7d;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(56, 239, 125, 0.3);
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Enhanced metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #4facfe;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)


def render_header():
    """Render the modern animated header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced AI Skill Gap Analyzer</h1>
        <p>BERT-Powered Semantic Skill Matching & Gap Analysis - Milestone 3</p>
    </div>
    """, unsafe_allow_html=True)


def create_skill_tags_html(skills):
    """Create HTML for skill tags with modern styling"""
    if not skills:
        return "<p style='color: #999;'>No skills detected</p>"
    
    html = '<div style="margin-top: 1rem;">'
    for skill in sorted(list(skills))[:50]:
        html += f'<span class="skill-tag">{skill}</span>'
    if len(skills) > 50:
        html += f'<span class="skill-tag" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">+{len(skills)-50} more</span>'
    html += '</div>'
    return html


@dataclass
class SkillMatch:
    """Data class for skill match information"""
    jd_skill: str
    resume_skill: str
    similarity: float
    category: str
    confidence_level: str
    priority: str = "MEDIUM"
    
    def to_dict(self) -> Dict:
        return {
            'jd_skill': self.jd_skill,
            'resume_skill': self.resume_skill,
            'similarity': self.similarity,
            'category': self.category,
            'confidence_level': self.confidence_level,
            'priority': self.priority
        }


@dataclass
class GapAnalysisResult:
    """Complete gap analysis results"""
    matched_skills: List[SkillMatch]
    partial_matches: List[SkillMatch]
    missing_skills: List[SkillMatch]
    overall_score: float
    category_scores: Dict[str, float]
    similarity_matrix: np.ndarray
    resume_skills: List[str]
    jd_skills: List[str]
    
    def get_statistics(self) -> Dict:
        total = len(self.jd_skills)
        return {
            'total_required_skills': total,
            'matched_count': len(self.matched_skills),
            'partial_count': len(self.partial_matches),
            'missing_count': len(self.missing_skills),
            'match_percentage': (len(self.matched_skills) / total * 100) if total > 0 else 0,
            'overall_score': self.overall_score * 100
        }


class SentenceBERTEncoder:
    """Handles BERT embedding generation using Sentence-BERT"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.logger = self._setup_logger()
        self.embedding_cache = {}
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_skills(self, skills: List[str], use_cache: bool = True, 
                     show_progress: bool = False) -> np.ndarray:
        if not skills:
            raise ValueError("Skills list cannot be empty")
        
        if use_cache:
            cached_embeddings = []
            uncached_skills = []
            uncached_indices = []
            
            for i, skill in enumerate(skills):
                if skill in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[skill])
                else:
                    uncached_skills.append(skill)
                    uncached_indices.append(i)
            
            if uncached_skills:
                new_embeddings = self.model.encode(
                    uncached_skills, 
                    show_progress_bar=show_progress,
                    batch_size=32
                )
                
                for skill, embedding in zip(uncached_skills, new_embeddings):
                    self.embedding_cache[skill] = embedding
                
                all_embeddings = [None] * len(skills)
                cached_idx = 0
                uncached_idx = 0
                
                for i in range(len(skills)):
                    if i in uncached_indices:
                        all_embeddings[i] = new_embeddings[uncached_idx]
                        uncached_idx += 1
                    else:
                        all_embeddings[i] = cached_embeddings[cached_idx]
                        cached_idx += 1
                
                return np.array(all_embeddings)
            else:
                return np.array(cached_embeddings)
        else:
            embeddings = self.model.encode(
                skills, 
                show_progress_bar=show_progress,
                batch_size=32
            )
            return embeddings
    
    def clear_cache(self):
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('BERTEncoder')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SimilarityCalculator:
    """Compute similarity scores between skills"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def compute_similarity_matrix(self, resume_embeddings: np.ndarray,
                                  jd_embeddings: np.ndarray) -> np.ndarray:
        self.logger.info(f"Computing similarity matrix: {resume_embeddings.shape} x {jd_embeddings.shape}")
        similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
        self.logger.info(f"Similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('SimilarityCalculator')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class SkillGapAnalyzer:
    """Main skill gap analysis engine"""
    
    def __init__(self, encoder: SentenceBERTEncoder, calculator: SimilarityCalculator,
                 strong_threshold: float = 0.80, partial_threshold: float = 0.50):
        self.encoder = encoder
        self.calculator = calculator
        self.strong_threshold = strong_threshold
        self.partial_threshold = partial_threshold
        self.logger = self._setup_logger()
    
    def analyze(self, resume_skills: List[str], jd_skills: List[str],
               skill_categories: Optional[Dict[str, str]] = None) -> GapAnalysisResult:
        self.logger.info(f"Starting gap analysis: {len(resume_skills)} resume skills vs {len(jd_skills)} JD skills")
        
        if not resume_skills or not jd_skills:
            raise ValueError("Both resume_skills and jd_skills must be non-empty")
        
        self.logger.info("Step 1: Generating BERT embeddings...")
        resume_embeddings = self.encoder.encode_skills(resume_skills, show_progress=True)
        jd_embeddings = self.encoder.encode_skills(jd_skills, show_progress=True)
        
        self.logger.info("Step 2: Computing similarity matrix...")
        similarity_matrix = self.calculator.compute_similarity_matrix(
            resume_embeddings, 
            jd_embeddings
        )
        
        self.logger.info("Step 3: Classifying skill matches...")
        matched_skills = []
        partial_matches = []
        missing_skills = []
        
        for jd_idx, jd_skill in enumerate(jd_skills):
            best_resume_idx = np.argmax(similarity_matrix[:, jd_idx])
            best_similarity = float(similarity_matrix[best_resume_idx, jd_idx])
            resume_skill = resume_skills[best_resume_idx]
            
            category = skill_categories.get(jd_skill, 'other') if skill_categories else 'other'
            
            if best_similarity >= self.strong_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='STRONG_MATCH',
                    confidence_level='HIGH',
                    priority='LOW'
                )
                matched_skills.append(match)
                
            elif best_similarity >= self.partial_threshold:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='PARTIAL_MATCH',
                    confidence_level='MEDIUM',
                    priority='MEDIUM'
                )
                partial_matches.append(match)
                
            else:
                match = SkillMatch(
                    jd_skill=jd_skill,
                    resume_skill=resume_skill,
                    similarity=best_similarity,
                    category='MISSING',
                    confidence_level='LOW',
                    priority='HIGH'
                )
                missing_skills.append(match)
        
        overall_score = self._calculate_overall_score(similarity_matrix)
        category_scores = self._calculate_category_scores(
            matched_skills, partial_matches, missing_skills
        )
        
        self.logger.info(f"Analysis complete: {len(matched_skills)} matched, "
                        f"{len(partial_matches)} partial, {len(missing_skills)} missing")
        
        return GapAnalysisResult(
            matched_skills=matched_skills,
            partial_matches=partial_matches,
            missing_skills=missing_skills,
            overall_score=overall_score,
            category_scores=category_scores,
            similarity_matrix=similarity_matrix,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
    
    def _calculate_overall_score(self, similarity_matrix: np.ndarray) -> float:
        max_similarities = similarity_matrix.max(axis=0)
        overall_score = float(np.mean(max_similarities))
        return overall_score
    
    def _calculate_category_scores(self, matched: List[SkillMatch],
                                   partial: List[SkillMatch],
                                   missing: List[SkillMatch]) -> Dict[str, float]:
        category_scores = {}
        all_skills = matched + partial + missing
        categories = set(skill.category for skill in all_skills)
        
        for category in categories:
            cat_skills = [s for s in all_skills if s.category == category]
            if cat_skills:
                avg_similarity = np.mean([s.similarity for s in cat_skills])
                category_scores[category] = float(avg_similarity)
        
        return category_scores
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('SkillGapAnalyzer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class GapVisualizer:
    """Create modern visualizations for gap analysis"""
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: np.ndarray,
                                 resume_skills: List[str],
                                 jd_skills: List[str]) -> go.Figure:
        max_display = 20
        display_resume = resume_skills[:max_display]
        display_jd = jd_skills[:max_display]
        display_matrix = similarity_matrix[:max_display, :max_display]
        
        fig = go.Figure(data=go.Heatmap(
            z=display_matrix,
            x=display_jd,
            y=display_resume,
            colorscale='RdYlGn',
            zmid=0.5,
            text=np.round(display_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Similarity",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.2
            )
        ))
        
        fig.update_layout(
            title=f"Skill Similarity Heatmap",
            xaxis_title="Job Description Skills",
            yaxis_title="Resume Skills",
            height=600,
            width=900,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        return fig
    
    @staticmethod
    def create_match_distribution_pie(analysis_result: GapAnalysisResult) -> go.Figure:
        stats = analysis_result.get_statistics()
        
        labels = ['Strong Matches', 'Partial Matches', 'Missing Skills']
        values = [stats['matched_count'], stats['partial_count'], stats['missing_count']]
        colors = ['#38ef7d', '#f093fb', '#ff6b6b']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            textposition='auto',
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white')
        )])
        
        fig.update_layout(
            title="Skill Match Distribution",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_overall_score_gauge(overall_score: float) -> go.Figure:
        score_percentage = overall_score * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Match Score", 'font': {'size': 24, 'color': 'white'}},
            delta={'reference': 70, 'increasing': {'color': "#38ef7d"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#667eea"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.3)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 107, 107, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(255, 165, 2, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(56, 239, 125, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#ff6b6b", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            number={'font': {'color': 'white', 'size': 40}}
        ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig


class ReportGenerator:
    """Generate comprehensive reports"""
    
    def __init__(self):
        self.timestamp = datetime.now()
    
    def generate_text_report(self, analysis_result: GapAnalysisResult) -> str:
        stats = analysis_result.get_statistics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SKILL GAP ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("-" * 80)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Overall Match Score: {stats['overall_score']:.1f}%")
        report_lines.append(f"Total Required Skills: {stats['total_required_skills']}")
        report_lines.append(f"Matched Skills: {stats['matched_count']} ({stats['match_percentage']:.1f}%)")
        report_lines.append(f"Partial Matches: {stats['partial_count']}")
        report_lines.append(f"Missing Skills: {stats['missing_count']}")
        report_lines.append("")
        
        if analysis_result.matched_skills:
            report_lines.append("-" * 80)
            report_lines.append("✓ STRONG MATCHES (Similarity ≥ 80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.matched_skills:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Resume: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append("")
        
        if analysis_result.partial_matches:
            report_lines.append("-" * 80)
            report_lines.append("⚠ PARTIAL MATCHES (Similarity 50-80%)")
            report_lines.append("-" * 80)
            for match in analysis_result.partial_matches:
                report_lines.append(f"  • {match.jd_skill}")
                report_lines.append(f"    Closest: {match.resume_skill}")
                report_lines.append(f"    Similarity: {match.similarity*100:.1f}%")
                report_lines.append(f"    Recommendation: Strengthen knowledge in {match.jd_skill}")
                report_lines.append("")
        
        if analysis_result.missing_skills:
            report_lines.append("-" * 80)
            report_lines.append("✗ CRITICAL GAPS (Similarity < 50%)")
            report_lines.append("-" * 80)
            for match in analysis_result.missing_skills:
                report_lines.append(f"  • {match.jd_skill} - {match.priority} PRIORITY")
                report_lines.append(f"    Current closest: {match.resume_skill} ({match.similarity*100:.1f}%)")
                report_lines.append(f"    Action: Acquire {match.jd_skill} through training/certification")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_json_report(self, analysis_result: GapAnalysisResult) -> str:
        stats = analysis_result.get_statistics()
        
        report_data = {
            'timestamp': self.timestamp.isoformat(),
            'statistics': stats,
            'matched_skills': [match.to_dict() for match in analysis_result.matched_skills],
            'partial_matches': [match.to_dict() for match in analysis_result.partial_matches],
            'missing_skills': [match.to_dict() for match in analysis_result.missing_skills],
            'category_scores': analysis_result.category_scores,
            'resume_skills': analysis_result.resume_skills,
            'jd_skills': analysis_result.jd_skills
        }
        
        return json.dumps(report_data, indent=2)


class CompleteSkillGapApp:
    """Complete Milestone 3 Application with Modern UI"""
    
    def __init__(self):
        self.encoder = SentenceBERTEncoder()
        self.calculator = SimilarityCalculator()
        self.visualizer = GapVisualizer()
        self.report_generator = ReportGenerator()
        
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'resume_skills' not in st.session_state:
            st.session_state.resume_skills = []
        if 'jd_skills' not in st.session_state:
            st.session_state.jd_skills = []
    
    def run(self):
        render_header()
        
        tabs = st.tabs([
            "🔍 Gap Analysis",
            "📊 Visualizations",
            "📈 Similarity Matrix",
            "📥 Export Reports",
            "⚙️ Settings"
        ])
        
        with tabs[0]:
            self._gap_analysis_tab()
        
        with tabs[1]:
            self._visualizations_tab()
        
        with tabs[2]:
            self._similarity_matrix_tab()
        
        with tabs[3]:
            self._export_tab()
        
        with tabs[4]:
            self._settings_tab()
    
    def _gap_analysis_tab(self):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Skill Gap Analysis")
        
        st.markdown("""
        **Powered by BERT Semantic Understanding:**
        1. Enter skills from resume and job description
        2. System generates semantic embeddings
        3. Computes cosine similarity between skills
        4. Identifies matches, partial matches, and gaps
        5. Provides actionable recommendations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Sample Data"],
            horizontal=True
        )
        
        resume_skills = []
        jd_skills = []
        
        if input_method == "Manual Entry":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📄 Resume Skills")
                resume_text = st.text_area(
                    "Enter skills (one per line):",
                    height=300,
                    placeholder="Python\nMachine Learning\nSQL\nData Analysis",
                    key="resume_input"
                )
                if resume_text:
                    resume_skills = [s.strip() for s in resume_text.split('\n') if s.strip()]
                    st.info(f"**{len(resume_skills)} skills entered**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("💼 Job Description Skills")
                jd_text = st.text_area(
                    "Enter required skills (one per line):",
                    height=300,
                    placeholder="Python\nDeep Learning\nTensorFlow\nSQL\nAWS",
                    key="jd_input"
                )
                if jd_text:
                    jd_skills = [s.strip() for s in jd_text.split('\n') if s.strip()]
                    st.info(f"**{len(jd_skills)} skills entered**")
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # Sample Data
            st.info("Using sample data for demonstration")
            resume_skills = [
                "Python", "Machine Learning", "SQL", "Data Analysis",
                "Pandas", "NumPy", "Scikit-learn", "Git", "Statistics",
                "Data Visualization", "Excel", "R Programming"
            ]
            jd_skills = [
                "Python", "Deep Learning", "TensorFlow", "SQL",
                "AWS", "Docker", "Kubernetes", "Data Science",
                "Neural Networks", "Cloud Computing", "PyTorch", "MLOps"
            ]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.success(f"✅ Sample Resume: {len(resume_skills)} skills")
                st.markdown(create_skill_tags_html(resume_skills), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.success(f"✅ Sample JD: {len(jd_skills)} skills")
                st.markdown(create_skill_tags_html(jd_skills), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Skill Gaps", type="primary", use_container_width=True):
                if not resume_skills or not jd_skills:
                    st.error("⚠️ Please provide both resume and JD skills")
                else:
                    self._perform_analysis(resume_skills, jd_skills)
        
        if st.session_state.analysis_result:
            st.markdown("---")
            self._display_analysis_results(st.session_state.analysis_result)
    
    def _perform_analysis(self, resume_skills: List[str], jd_skills: List[str]):
        with st.spinner("🔄 Analyzing skills..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing analyzer...")
                progress_bar.progress(20)
                
                strong_threshold = st.session_state.get('strong_threshold', 0.80)
                partial_threshold = st.session_state.get('partial_threshold', 0.50)
                
                analyzer = SkillGapAnalyzer(
                    self.encoder,
                    self.calculator,
                    strong_threshold=strong_threshold,
                    partial_threshold=partial_threshold
                )
                
                status_text.text("Running gap analysis...")
                progress_bar.progress(40)
                
                result = analyzer.analyze(resume_skills, jd_skills)
                
                progress_bar.progress(80)
                status_text.text("Storing results...")
                
                st.session_state.analysis_result = result
                st.session_state.resume_skills = resume_skills
                st.session_state.jd_skills = jd_skills
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                st.success("✅ Gap analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _display_analysis_results(self, result: GapAnalysisResult):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📊 Analysis Results")
        
        stats = result.get_statistics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Overall Score", f"{stats['overall_score']:.1f}%")
        with col2:
            st.metric("Total Required", stats['total_required_skills'])
        with col3:
            st.metric("✅ Matched", stats['matched_count'])
        with col4:
            st.metric("⚠️ Partial", stats['partial_count'])
        with col5:
            st.metric("❌ Missing", stats['missing_count'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if result.matched_skills:
            with st.expander(f"✅ **Strong Matches ({len(result.matched_skills)})**", expanded=True):
                for match in result.matched_skills:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f'<div class="strong-match">**{match.jd_skill}** ↔ {match.resume_skill}</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.metric("Similarity", f"{match.similarity*100:.1f}%")
        
        if result.partial_matches:
            with st.expander(f"⚠️ **Partial Matches ({len(result.partial_matches)})**"):
                for match in result.partial_matches:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f'<div class="partial-match">**{match.jd_skill}** ↔ {match.resume_skill}</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"💡 Recommendation: Strengthen knowledge in {match.jd_skill}")
                    with col2:
                        st.metric("Similarity", f"{match.similarity*100:.1f}%")
        
        if result.missing_skills:
            with st.expander(f"❌ **Missing Skills ({len(result.missing_skills)})**"):
                for match in result.missing_skills:
                    priority_class = f"priority-{match.priority.lower()}"
                    st.markdown(
                        f'<div class="missing-skill">**{match.jd_skill}** - <span class="{priority_class}">{match.priority} PRIORITY</span></div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"Current closest: {match.resume_skill} ({match.similarity*100:.1f}%)")
    
    def _visualizations_tab(self):
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📊 Visual Analytics")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Overall Match Score")
        fig_gauge = self.visualizer.create_overall_score_gauge(result.overall_score)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Match Distribution")
            fig_pie = self.visualizer.create_match_distribution_pie(result)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Skills Breakdown")
            stats = result.get_statistics()
            st.markdown(f"""
            - **Total Required:** {stats['total_required_skills']}
            - **Match Rate:** {stats['match_percentage']:.1f}%
            - **Strong Matches:** {stats['matched_count']}
            - **Partial Matches:** {stats['partial_count']}
            - **Gaps:** {stats['missing_count']}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _similarity_matrix_tab(self):
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📈 Similarity Matrix Analysis")
        
        st.markdown("""
        This heatmap shows the semantic similarity between all resume skills and job description skills.
        - **Green**: High similarity (strong match)
        - **Yellow**: Medium similarity (partial match)
        - **Red**: Low similarity (skill gap)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        fig_heatmap = self.visualizer.create_similarity_heatmap(
            result.similarity_matrix,
            result.resume_skills,
            result.jd_skills
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with st.expander("📋 View Detailed Similarity Matrix"):
            df_matrix = pd.DataFrame(
                result.similarity_matrix,
                index=result.resume_skills,
                columns=result.jd_skills
            )
            
            df_display = df_matrix.applymap(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(df_display, use_container_width=True)
            
            csv_matrix = df_matrix.to_csv()
            st.download_button(
                "📥 Download Similarity Matrix (CSV)",
                csv_matrix,
                f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    def _export_tab(self):
        if not st.session_state.analysis_result:
            st.info("👈 Please run gap analysis first in the 'Gap Analysis' tab")
            return
        
        result = st.session_state.analysis_result
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📥 Export Analysis Reports")
        
        st.markdown("""
        Download your skill gap analysis in various formats:
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📄 Text Report")
            st.markdown("Comprehensive text report with all details")
            
            text_report = self.report_generator.generate_text_report(result)
            
            st.download_button(
                label="📥 Download TXT",
                data=text_report,
                file_name=f"skill_gap_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📋 JSON Report")
            st.markdown("Structured data for integration")
            
            json_report = self.report_generator.generate_json_report(result)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_report,
                file_name=f"skill_gap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 CSV Export")
            st.markdown("Skills data in spreadsheet format")
            
            # Create CSV
            data = []
            for match in result.matched_skills + result.partial_matches + result.missing_skills:
                data.append({
                    'JD Skill': match.jd_skill,
                    'Resume Skill': match.resume_skill,
                    'Similarity': f"{match.similarity*100:.2f}%",
                    'Category': match.category,
                    'Priority': match.priority
                })
            
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"skill_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📖 Report Preview")
        
        preview_tab = st.selectbox(
            "Select report to preview:",
            ["Text Report", "JSON Report"]
        )
        
        if preview_tab == "Text Report":
            st.text_area("Report Preview", text_report, height=400)
        else:
            st.json(json.loads(json_report))
    
    def _settings_tab(self):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("⚙️ Settings & Configuration")
        
        st.subheader("🎚️ Similarity Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strong_threshold = st.slider(
                "Strong Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('strong_threshold', 0.80),
                step=0.05,
                help="Minimum similarity for a skill to be considered a strong match"
            )
            st.session_state.strong_threshold = strong_threshold
        
        with col2:
            partial_threshold = st.slider(
                "Partial Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('partial_threshold', 0.50),
                step=0.05,
                help="Minimum similarity for a skill to be considered a partial match"
            )
            st.session_state.partial_threshold = partial_threshold
        
        st.info(f"""
        **Current Configuration:**
        - Strong Match: Similarity ≥ {strong_threshold:.0%}
        - Partial Match: {partial_threshold:.0%} ≤ Similarity < {strong_threshold:.0%}
        - Missing/Gap: Similarity < {partial_threshold:.0%}
        """)
        
        st.markdown("---")
        st.subheader("🤖 Model Configuration")
        
        st.info(f"""
        **Current Model:** {self.encoder.model_name}
        **Embedding Dimension:** {self.encoder.embedding_dimension}
        **Cache Size:** {len(self.encoder.embedding_cache)} embeddings
        """)
        
        if st.button("🗑️ Clear Embedding Cache"):
            self.encoder.clear_cache()
            st.success("Cache cleared!")
        
        st.markdown("---")
        st.subheader("ℹ️ About Milestone 3")
        
        st.markdown("""
        **Milestone 3: Skill Gap Analysis & Similarity Matching**
        
        **Features Implemented:**
        - ✅ BERT-based semantic similarity using Sentence-BERT
        - ✅ Cosine similarity computation
        - ✅ Multi-level skill gap identification
        - ✅ Interactive similarity matrices
        - ✅ Comprehensive visualizations
        - ✅ Multiple export formats
        - ✅ Modern, responsive UI design
        
        **Technologies Used:**
        - Sentence-Transformers (BERT)
        - Scikit-learn (Cosine Similarity)
        - Plotly (Interactive Visualizations)
        - Streamlit (Web Interface)
        
        **Model Details:**
        - Model: all-MiniLM-L6-v2
        - Embedding Size: 384 dimensions
        - Performance: Fast inference, good accuracy
        - Use Case: Semantic text similarity
        """)
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    try:
        app = CompleteSkillGapApp()
        app.run()
        
        with st.sidebar:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.header("🎯 Milestone 3")
            st.markdown("**Skill Gap Analysis**")
            
            st.markdown("---")
            st.subheader("📊 Quick Stats")
            
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                stats = result.get_statistics()
                
                st.metric("Overall Match", f"{stats['overall_score']:.1f}%")
                st.metric("Skills Analyzed", stats['total_required_skills'])
                
                st.markdown("**Breakdown:**")
                st.success(f"✅ Matched: {stats['matched_count']}")
                st.warning(f"⚠️ Partial: {stats['partial_count']}")
                st.error(f"❌ Missing: {stats['missing_count']}")
            else:
                st.info("No analysis yet. Start in the Gap Analysis tab!")
            
            st.markdown("---")
            st.subheader("🚀 Quick Actions")
            
            if st.button("🔄 Reset Analysis", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.resume_skills = []
                st.session_state.jd_skills = []
                st.rerun()
            
            st.markdown("---")
            st.caption("Milestone 3 - Enhanced UI")
            st.caption("Version 2.0.0")
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()