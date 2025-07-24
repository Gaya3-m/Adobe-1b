#!/usr/bin/env python3
"""
Enhanced Persona-Driven Document Intelligence System
with Hierarchical Understanding and Semantic Analysis
"""

import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse
from collections import defaultdict
import string

import PyPDF2
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from sklearn.cluster import AgglomerativeClustering

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass
class DocumentSection:
    document: str
    page_number: int
    section_title: str
    content: str
    importance_rank: int = 0
    section_level: str = "section"  # chapter, section, subsection

@dataclass
class SubSection:
    document: str
    page_number: int
    refined_text: str
    importance_rank: int = 0
    semantic_score: float = 0.0

class SemanticProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyphrase_model = KeyBERT()
        self.cache = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Cache embeddings for efficiency"""
        if text not in self.cache:
            self.cache[text] = self.embedding_model.encode(text)
        return self.cache[text]

    def extract_keyphrases(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        return self.keyphrase_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )

    def cluster_texts(self, texts: List[str], n_clusters: Optional[int] = None) -> Dict:
        """Hierarchical clustering of related content"""
        embeddings = self.embedding_model.encode(texts)
        
        if n_clusters is None:
            n_clusters = min(5, len(texts))  # Default to 5 or less
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            linkage='average'
        ).fit(embeddings)
        
        return {
            'labels': clustering.labels_,
            'texts': texts,
            'embeddings': embeddings.tolist()  # Convert to list for JSON serialization
        }

class DocumentAnalyzer:
    def __init__(self):
        self.semantic = SemanticProcessor()
        self.generic_titles = [
            "introduction", "overview", "background", 
            "conclusion", "summary", "abstract"
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page numbers"""
        page_texts = {}
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        page_texts[page_num] = text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return page_texts

    def identify_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """Hierarchical section detection with level identification"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        sections = []
        current_section = ("Document Start", [], "chapter")
        
        for line in lines:
            # Detect section headers
            level, title = self._detect_section_header(line)
            if title:
                if current_section[1]:  # Save previous section
                    sections.append((
                        current_section[0],
                        '\n'.join(current_section[1]),
                        current_section[2]
                    ))
                current_section = (title, [], level)
            else:
                current_section[1].append(line)
        
        if current_section[1]:
            sections.append((
                current_section[0],
                '\n'.join(current_section[1]),
                current_section[2]
            ))
        return sections

    def _detect_section_header(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Determine if line is a section header and its level"""
        # Chapter-level (1, 1.1, etc.)
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return "chapter", line
        
        # Section-level (CAPS or Title Case)
        if (re.match(r'^[A-Z][A-Z\s]{4,}$', line) or 
            (line[0].isupper() and len(line) < 80 and not line.endswith(('.', ':')))):
            return "section", line
        
        # Subsection-level (bold, italics, etc.)
        if re.match(r'^[_*\-#]{2,}\s*[A-Z]', line):
            return "subsection", line.strip('_*-# ')
            
        return None, None

    def create_persona_profile(self, persona: str, job: str) -> Dict[str, Any]:
        """Dynamic profile with semantic understanding"""
        profile_text = f"{persona}. {job}"
        embedding = self.semantic.get_embedding(profile_text)
        keyphrases = self.semantic.extract_keyphrases(profile_text)
        
        return {
            'embedding': embedding.tolist(),  # Convert numpy array to list
            'keyphrases': [kp[0] for kp in keyphrases],
            'content_weights': self._detect_content_weights(job),
            'context': self._extract_context(persona, job)
        }

    def _detect_content_weights(self, job: str) -> Dict[str, float]:
        """Detect needed content types from job description"""
        weights = {
            'conceptual': len(re.findall(r'\b(understand|learn|concept)\b', job.lower())),
            'procedural': len(re.findall(r'\b(method|process|steps)\b', job.lower())),
            'quantitative': len(re.findall(r'\b(analyze|metric|data)\b', job.lower())),
            'comparative': len(re.findall(r'\b(compare|versus|contrast)\b', job.lower())),
            'temporal': len(re.findall(r'\b(timeline|history|evolution)\b', job.lower()))
        }
        total = sum(weights.values()) or 1
        return {k: v/total for k, v in weights.items()}

    def _extract_context(self, persona: str, job: str) -> Dict[str, Any]:
        """Extract contextual clues from persona/job"""
        text = f"{persona} {job}".lower()
        return {
            'is_group': any(w in text for w in ['group', 'team', 'friends']),
            'is_student': any(w in text for w in ['student', 'college', 'university']),
            'timeframe': re.search(r'(\d+)\s*(day|week|month)', text),
            'budget': 'budget' in text or 'affordable' in text
        }

    def calculate_section_score(self, section: Tuple[str, str, str], profile: Dict) -> float:
        """Multidimensional relevance scoring"""
        title, content, level = section
        full_text = f"{title}. {content}"
        
        # 1. Semantic similarity
        embedding = self.semantic.get_embedding(full_text)
        semantic_score = float(cosine_similarity(
            np.array(profile['embedding']).reshape(1, -1),  # Convert back to numpy array
            embedding.reshape(1, -1)
        )[0][0])
        
        # 2. Keyphrase overlap
        keyphrase_score = sum(
            1 for kp in profile['keyphrases'] 
            if re.search(r'\b' + re.escape(kp) + r'\b', full_text.lower())
        ) / len(profile['keyphrases']) if profile['keyphrases'] else 0
        
        # 3. Content type alignment
        content_score = sum(
            weight * self._score_content_type(content.lower(), ctype)
            for ctype, weight in profile['content_weights'].items()
        )
        
        # 4. Contextual boosting
        context_boost = 1.0
        ctx = profile['context']
        if ctx['is_group'] and any(w in content.lower() for w in ['group', 'together']):
            context_boost *= 1.5
        if ctx['is_student'] and any(w in content.lower() for w in ['student', 'budget']):
            context_boost *= 1.3
        
        # 5. Level importance
        level_weight = {
            'chapter': 1.5,
            'section': 1.2,
            'subsection': 1.0
        }.get(level, 1.0)
        
        # Combine scores
        combined = (
            0.5 * semantic_score + 
            0.3 * keyphrase_score + 
            0.2 * content_score
        ) * context_boost * level_weight
        
        # Penalize generic sections
        if any(gt in title.lower() for gt in self.generic_titles):
            combined *= 0.7
            
        return float(min(max(combined, 0), 1))  # Ensure Python float

    def _score_content_type(self, text: str, content_type: str) -> float:
        """Score content based on type"""
        if content_type == 'conceptual':
            terms = ['defined as', 'refers to', 'concept of']
        elif content_type == 'procedural':
            terms = ['step', 'method', 'process', 'first', 'next']
        elif content_type == 'quantitative':
            terms = ['data', 'percent', 'statistic', 'figure']
        elif content_type == 'comparative':
            terms = ['vs', 'compared', 'than', 'relative']
        else:  # temporal
            terms = ['year', 'month', 'during', 'after']
        
        return min(sum(1 for term in terms if term in text) / len(terms), 1.0)

    def extract_subsections(self, content: str, profile: Dict, document: str, page_number: int, max_subs: int = 3) -> List[SubSection]:
        """Extract and rank relevant subsections"""
        # Split into meaningful chunks
        chunks = self._split_content(content)
        if not chunks:
            return []
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            embedding = self.semantic.get_embedding(chunk)
            score = float(cosine_similarity(
                np.array(profile['embedding']).reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0])
            scored_chunks.append((chunk, score))
        
        # Sort and select top
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [
            SubSection(
                document=document,
                page_number=page_number,
                refined_text=chunk,
                semantic_score=score,
                importance_rank=i+1
            )
            for i, (chunk, score) in enumerate(scored_chunks[:max_subs])
        ]

    def _split_content(self, text: str) -> List[str]:
        """Split content into meaningful chunks"""
        # Try paragraphs first
        chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Fallback to sentences if needed
        if len(chunks) < 2:
            chunks = nltk.sent_tokenize(text)
            chunks = [s.strip() for s in chunks if len(s.strip()) > 30]
        
        return chunks

    def process_documents(self, document_paths: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        start_time = time.time()
        profile = self.create_persona_profile(persona, job)
        
        # Process all documents
        all_sections = []
        doc_section_counts = defaultdict(int)
        
        for doc_path in document_paths:
            doc_name = Path(doc_path).name
            page_texts = self.extract_text_from_pdf(doc_path)
            
            for page_num, text in page_texts.items():
                sections = self.identify_sections(text)
                for title, content, level in sections:
                    score = self.calculate_section_score((title, content, level), profile)
                    section = DocumentSection(
                        document=doc_name,
                        page_number=page_num,
                        section_title=title,
                        content=content,
                        section_level=level
                    )
                    all_sections.append((section, score))
        
        # Rank sections (with diversity)
        all_sections.sort(key=lambda x: x[1], reverse=True)
        final_sections = []
        final_subsections = []
        doc_counts = defaultdict(int)
        
        for rank, (section, score) in enumerate(all_sections, 1):
            if doc_counts[section.document] >= 2:  # Max 2 sections per doc
                continue
                
            section.importance_rank = rank
            final_sections.append(section)
            doc_counts[section.document] += 1
            
            # Extract subsections
            subsections = self.extract_subsections(
                section.content, 
                profile,
                section.document,
                section.page_number
            )
            for sub in subsections:
                final_subsections.append(sub)
            
            if len(final_sections) >= 20:  # Limit total sections
                break
        
        # Prepare output with type conversion
        return {
            "metadata": {
                "input_documents": [Path(p).name for p in document_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": float(time.time() - start_time)
            },
            "extracted_sections": [
                {
                    "document": s.document,
                    "section_title": s.section_title,
                    "page_number": s.page_number,
                    "importance_rank": s.importance_rank,
                    "section_level": s.section_level
                }
                for s in final_sections
            ],
            "subsection_analysis": [
                {
                    "document": s.document,
                    "page_number": s.page_number,
                    "refined_text": s.refined_text,
                    "importance_rank": s.importance_rank,
                    "semantic_score": float(s.semantic_score)  # Convert numpy float to Python float
                }
                for s in sorted(final_subsections, key=lambda x: x.semantic_score, reverse=True)[:30]
            ]
        }

def load_config(input_json: str) -> dict:
    """Load and validate configuration"""
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract documents
        documents = []
        for doc in config.get('documents', []):
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, dict) and 'filename' in doc:
                documents.append(doc['filename'])
        
        # Extract persona
        persona = config.get('persona', '')
        if isinstance(persona, dict):
            persona = persona.get('role', '')
        
        # Extract job
        job = config.get('job_to_be_done', '')
        if isinstance(job, dict):
            job = job.get('task', '')
        
        return {
            'documents': documents,
            'persona': persona,
            'job': job,
            'config': config
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Enhanced Document Intelligence System")
    parser.add_argument("--input_json", required=True, help="Input configuration JSON")
    parser.add_argument("--output", default="output.json", help="Output file path")
    args = parser.parse_args()

    try:
        config = load_config(args.input_json)
        analyzer = DocumentAnalyzer()
        result = analyzer.process_documents(
            config['documents'],
            config['persona'],
            config['job']
        )
        
        # Add original config to metadata
        result['metadata']['config'] = config['config']
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Success! Results saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()