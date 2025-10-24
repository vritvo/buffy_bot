#!/usr/bin/env python3
"""
Static Site Generator for Conference Papers

Generates a static website from conference paper folders created by run-conference.
Creates a landing page, technical documentation, and pages for each accepted paper
with their reviews.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import click
from rich.console import Console
import shutil
import markdown

console = Console()


class PaperFolder:
    """Represents a paper folder with metadata and file paths"""
    
    def __init__(self, folder_path: Path):
        self.path = folder_path
        self.folder_name = folder_path.name
        self.timestamp, self.base_name = self._parse_folder_name()
        self.version = self._extract_version()
        self.paper_md = self._find_paper_md()
        self.paper_pdf = self._find_paper_pdf()
        self.reviews = self._find_reviews()
        self.title = self._extract_title()
        self.abstract = self._extract_abstract()
        
    def _parse_folder_name(self) -> Tuple[str, str]:
        """Parse folder name into timestamp and base name"""
        match = re.match(r'(\d{8}_\d{6})_(.+)', self.folder_name)
        if match:
            return match.group(1), match.group(2)
        return "", self.folder_name
    
    def _extract_version(self) -> int:
        """Extract version number from base name (e.g., _v2_v2 -> 2)"""
        version_count = self.base_name.count('_v')
        return version_count
    
    def _get_base_paper_name(self) -> str:
        """Get the base paper name without version suffixes"""
        # Remove all _v2, _v2_v2 type suffixes
        base = re.sub(r'(_v\d+)+$', '', self.base_name)
        return base
    
    def _find_paper_md(self) -> Optional[Path]:
        """Find the paper markdown file"""
        # Look for paper files in various naming patterns
        patterns = ['paper*.md', '*.md']
        for pattern in patterns:
            matches = list(self.path.glob(pattern))
            if matches:
                # Prefer files with 'paper' in the name
                paper_files = [m for m in matches if 'paper' in m.name.lower()]
                if paper_files:
                    return paper_files[0]
                return matches[0]
        return None
    
    def _find_paper_pdf(self) -> Optional[Path]:
        """Find the paper PDF file (prioritize scanned versions ending in _scan.pdf)"""
        # First, try to find scanned PDFs (aged versions)
        scan_matches = list(self.path.glob('*_scan.pdf'))
        if scan_matches:
            return scan_matches[0]
        
        # If no scanned PDFs found, return None (only use scanned versions for the conference site)
        return None
    
    def _find_reviews(self) -> List[Path]:
        """Find all review JSON files"""
        reviews_dir = self.path / 'reviews'
        if reviews_dir.exists():
            return sorted(reviews_dir.glob('*.json'))
        return []
    
    def _extract_title(self) -> str:
        """Extract title from paper markdown frontmatter"""
        if not self.paper_md or not self.paper_md.exists():
            return self._humanize_title(self.base_name)
        
        try:
            with open(self.paper_md, 'r') as f:
                content = f.read()
                # Look for YAML frontmatter
                if content.startswith('---'):
                    frontmatter_end = content.find('---', 3)
                    if frontmatter_end > 0:
                        frontmatter = content[3:frontmatter_end]
                        for line in frontmatter.split('\n'):
                            if line.startswith('title:'):
                                title = line.split('title:', 1)[1].strip().strip('"\'')
                                return title
                
                # Look for first heading
                for line in content.split('\n'):
                    if line.startswith('# '):
                        return line.lstrip('# ').strip()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not extract title from {self.paper_md}: {e}[/yellow]")
        
        return self._humanize_title(self.base_name)
    
    def _humanize_title(self, text: str) -> str:
        """Convert folder name to readable title"""
        # Remove version suffixes
        text = re.sub(r'(_v\d+)+$', '', text)
        # Replace underscores with spaces
        text = text.replace('_', ' ')
        return text
    
    def _extract_abstract(self) -> str:
        """Extract abstract from paper markdown"""
        if not self.paper_md or not self.paper_md.exists():
            return ""
        
        try:
            with open(self.paper_md, 'r') as f:
                content = f.read()
                
                # Skip frontmatter
                if content.startswith('---'):
                    frontmatter_end = content.find('---', 3)
                    if frontmatter_end > 0:
                        content = content[frontmatter_end + 3:]
                
                # Look for Abstract section
                abstract_pattern = r'#{1,3}\s*Abstract\s*\n\n(.*?)(?=\n#{1,3}|\Z)'
                match = re.search(abstract_pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    abstract = match.group(1).strip()
                    # Remove scratchpad if present
                    if '<scratchpad>' in abstract:
                        abstract = abstract.split('<scratchpad>')[0].strip()
                    return abstract
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not extract abstract from {self.paper_md}: {e}[/yellow]")
        
        return ""
    
    def get_base_paper_series(self) -> str:
        """Get the base paper series name (without versions)"""
        return re.sub(r'(_v\d+)+$', '', self.base_name)
    
    def is_accepted(self) -> bool:
        """Check if this paper has all reviews marked as ACCEPT"""
        if not self.reviews:
            return False
        
        try:
            for review_path in self.reviews:
                with open(review_path, 'r') as f:
                    review_data = json.load(f)
                    decision = review_data.get('metadata', {}).get('decision', '')
                    if decision != 'ACCEPT':
                        return False
            return True
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check acceptance status for {self.path}: {e}[/yellow]")
            return False


class StaticSiteGenerator:
    """Generates static HTML site from conference papers"""
    
    def __init__(self, papers_dir: Path, output_dir: Path, svg_file: Optional[Path] = None):
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.svg_file = svg_file
        self.svg_content: Optional[str] = None
        self.papers: List[PaperFolder] = []
        self.paper_series: Dict[str, List[PaperFolder]] = {}  # Track all versions
        
        # Load SVG content if provided
        if svg_file and svg_file.exists():
            self.svg_content = svg_file.read_text()
        
    def scan_papers(self):
        """Scan papers directory and identify final accepted versions"""
        console.print("[cyan]Scanning papers directory...[/cyan]")
        
        all_folders = [f for f in self.papers_dir.iterdir() if f.is_dir()]
        all_papers = [PaperFolder(f) for f in all_folders]
        
        # Group by base paper series
        for paper in all_papers:
            series = paper.get_base_paper_series()
            if series not in self.paper_series:
                self.paper_series[series] = []
            self.paper_series[series].append(paper)
        
        # For each series, find the highest version that's accepted
        for series, versions in self.paper_series.items():
            # Sort by version number (highest first)
            versions.sort(key=lambda p: p.version, reverse=True)
            
            # Find first accepted version
            for paper in versions:
                if paper.is_accepted():
                    self.papers.append(paper)
                    break
            else:
                # If no accepted version, take the latest version
                if versions:
                    console.print(f"[yellow]Warning: No accepted version found for '{series}', using latest version[/yellow]")
                    self.papers.append(versions[0])
        
        # Sort papers by title
        self.papers.sort(key=lambda p: p.title)
        
        console.print(f"[green]Found {len(self.papers)} final papers[/green]")
    
    def generate_site(self, landing_md: Optional[Path], tech_md: Optional[Path], custom_css: Optional[Path] = None):
        """Generate complete static site"""
        console.print("\n[cyan]Generating static site...[/cyan]")
        
        # Validate required markdown files
        if not landing_md or not landing_md.exists():
            console.print(f"[red]Error: Landing page markdown file not found: {landing_md}[/red]")
            raise FileNotFoundError(f"Landing page markdown file required but not found: {landing_md}")
        
        if not tech_md or not tech_md.exists():
            console.print(f"[red]Error: Technical documentation markdown file not found: {tech_md}[/red]")
            raise FileNotFoundError(f"Technical documentation markdown file required but not found: {tech_md}")
        
        # Validate custom CSS if provided
        if custom_css and not custom_css.exists():
            console.print(f"[red]Error: Custom CSS file not found: {custom_css}[/red]")
            raise FileNotFoundError(f"Custom CSS file not found: {custom_css}")
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'papers').mkdir(exist_ok=True)
        (self.output_dir / 'iterations').mkdir(exist_ok=True)
        (self.output_dir / 'pdfs').mkdir(exist_ok=True)
        (self.output_dir / 'css').mkdir(exist_ok=True)
        
        # Generate or copy CSS
        self._generate_css(custom_css)
        
        # Generate landing page
        self._generate_landing_page(landing_md)
        
        # Generate tech documentation
        self._generate_tech_page(tech_md)
        
        # Generate paper pages and copy all versions
        for paper in self.papers:
            # Get all versions for this paper series
            series_name = paper.get_base_paper_series()
            all_versions = self.paper_series.get(series_name, [paper])
            # Sort by version (lowest first for chronological order)
            all_versions.sort(key=lambda p: p.version)
            
            self._generate_paper_page(paper, all_versions)
            
            # Copy PDFs for all versions
            for version in all_versions:
                self._copy_paper_pdf(version)
            
            # Generate iteration pages
            self._generate_iteration_pages(paper, all_versions)
        
        console.print(f"\n[green]✓ Site generated successfully in {self.output_dir}[/green]")
    
    def _generate_css(self, custom_css: Optional[Path] = None):
        """Generate CSS stylesheet or copy custom CSS file
        
        Args:
            custom_css: Optional path to custom CSS file. If provided, copies this file
                       instead of generating the default CSS.
        """
        css_output_path = self.output_dir / 'css' / 'style.css'
        
        if custom_css:
            console.print(f"[cyan]Using custom CSS: {custom_css}[/cyan]")
            shutil.copy2(custom_css, css_output_path)
            return
        
        console.print("[cyan]Generating default CSS...[/cyan]")
        css = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background: white;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
}

.content {
    padding: 0;
    margin-bottom: 30px;
}

.content h1 {
    font-size: 2.5em;
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 3px solid #667eea;
}

.inline-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}

.inline-link:hover {
    color: #764ba2;
    text-decoration: underline;
}

.paper-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 30px;
    margin-top: 30px;
}

.paper-card {
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
    border-left: 4px solid #667eea;
}

.paper-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}

.paper-card h3 {
    color: #667eea;
    margin-bottom: 15px;
    font-size: 1.3em;
}

.paper-card .abstract {
    color: #666;
    font-size: 0.95em;
    margin-bottom: 15px;
    line-height: 1.6;
}

.paper-card a.read-more {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    display: inline-block;
    margin-top: 10px;
}

.paper-card a.read-more:hover {
    color: #764ba2;
}

.paper-details h2 {
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #667eea;
}

.paper-details .abstract {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 6px;
    margin: 20px 0;
    line-height: 1.8;
}

.links {
    margin: 30px 0;
    display: flex;
    flex-direction: column;
    gap: 15px;
    align-items: flex-start;
}

.btn {
    display: inline-block;
    padding: 12px 24px;
    background: #667eea;
    color: white;
    text-decoration: none;
    border-radius: 6px;
    font-weight: 500;
    transition: background 0.2s, transform 0.2s;
}

.btn:hover {
    background: #764ba2;
    transform: translateY(-2px);
}

.btn.secondary {
    background: #6c757d;
}

.btn.secondary:hover {
    background: #5a6268;
}

.reviews-list {
    margin-top: 30px;
}

.reviews-list h3 {
    color: #667eea;
    margin-bottom: 15px;
}

.review-item {
    background: #f8f9fa;
    padding: 15px 20px;
    margin-bottom: 10px;
    border-radius: 6px;
    border-left: 4px solid #667eea;
}

.review-item .decision {
    display: inline-block;
    padding: 4px 12px;
    background: #28a745;
    color: white;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: 600;
    margin-left: 10px;
}

.review-item .decision.accept {
    background: #28a745;
}

.review-item .decision.reject {
    background: #dc3545;
}

.iteration-info {
    color: #666;
    font-style: italic;
    margin-bottom: 20px;
}

.iteration-details h3 {
    color: #667eea;
    margin-top: 30px;
    margin-bottom: 15px;
}

.iteration-details h4 {
    color: #555;
    margin-top: 20px;
    margin-bottom: 10px;
}

.review-content {
    background: #f8f9fa;
    padding: 30px;
    border-radius: 8px;
    margin: 20px 0;
}

.review-content .decision {
    display: inline-block;
    padding: 4px 12px;
    color: white;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: 600;
}

.review-content .decision.accept {
    background: #28a745;
}

.review-content .decision.reject {
    background: #dc3545;
}

.review-content ul {
    margin-left: 20px;
    margin-bottom: 15px;
}

.review-content li {
    margin-bottom: 8px;
}

footer {
    text-align: center;
    padding: 40px 0;
    color: #666;
    margin-top: 60px;
    border-top: 1px solid #ddd;
}

.markdown-content h1 {
    font-size: 2.5em;
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 3px solid #667eea;
}

.markdown-content h2 {
    color: #667eea;
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 1.8em;
}

.markdown-content h3 {
    color: #555;
    margin-top: 25px;
    margin-bottom: 12px;
    font-size: 1.4em;
}

.markdown-content p {
    margin-bottom: 15px;
}

.markdown-content ul,
.markdown-content ol {
    margin-left: 30px;
    margin-bottom: 15px;
}

.markdown-content code {
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
}

.markdown-content pre {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    overflow-x: auto;
    margin-bottom: 15px;
}

.papers-heading {
    margin-top: 50px;
    margin-bottom: 30px;
    color: #667eea;
    font-size: 2em;
    padding-bottom: 10px;
    border-bottom: 2px solid #667eea;
}

.abstract-heading {
    color: #667eea;
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 1.8em;
}

.paper-details h1 {
    font-size: 2.5em;
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 3px solid #667eea;
}

.review-content h1 {
    font-size: 2.2em;
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 3px solid #667eea;
}

.review-content h2 {
    color: #667eea;
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 1.5em;
}

@media (max-width: 768px) {
    .paper-grid {
        grid-template-columns: 1fr;
    }
    
    .content h1,
    .markdown-content h1 {
        font-size: 2em;
    }
    
    .container {
        padding: 20px 15px;
    }
}
"""
        css_path = self.output_dir / 'css' / 'style.css'
        css_path.write_text(css)
    
    def _generate_landing_page(self, landing_md: Path):
        """Generate landing page"""
        console.print("[cyan]Generating landing page...[/cyan]")
        
        # Read landing page content and convert to HTML
        landing_content = markdown.markdown(landing_md.read_text())
        
        # Insert "Read more" link to technical documentation after first paragraph
        # Split content into paragraphs and insert link after first </p>
        first_p_end = landing_content.find('</p>')
        if first_p_end != -1:
            tech_link = '\n<p><a href="technical_documentation.html" class="inline-link">Read more</a></p>'
            landing_content = landing_content[:first_p_end] + landing_content[first_p_end:first_p_end+4] + tech_link + landing_content[first_p_end+4:]
        
        # Generate paper list
        papers_html = '<div class="paper-grid">'
        for paper in self.papers:
            # Convert abstract markdown to HTML
            abstract_html = markdown.markdown(paper.abstract)
            papers_html += f"""
            <div class="paper-card">
                <h3>{paper.title}</h3>
                <div class="abstract">{abstract_html}</div>
                <a href="papers/{paper.folder_name}.html" class="read-more">Read more →</a>
            </div>
            """
        papers_html += '</div>'
        
        html = self._wrap_html(
            title="Conference Papers",
            content=f"""
            <div class="markdown-content">
                {landing_content}
            </div>
            
            <h2 class="papers-heading">Accepted Papers</h2>
            {papers_html}
            """
        )
        
        (self.output_dir / 'index.html').write_text(html)
    
    def _generate_tech_page(self, tech_md: Path):
        """Generate technical documentation page"""
        console.print("[cyan]Generating technical documentation...[/cyan]")
        
        content = markdown.markdown(tech_md.read_text(), extensions=['fenced_code', 'tables'])
        
        html = self._wrap_html(
            title="Technical Documentation",
            content=f"""
            <div class="markdown-content">
                {content}
            </div>
            <div class="links">
                <a href="index.html" class="btn secondary">← Back to Conference</a>
            </div>
            """,
            css_path="css/style.css"
        )
        
        (self.output_dir / 'technical_documentation.html').write_text(html)
    
    def _generate_paper_page(self, paper: PaperFolder, all_versions: List[PaperFolder]):
        """Generate individual paper page with review iterations"""
        console.print(f"[cyan]Generating page for: {paper.title}[/cyan]")
        
        # Generate iterations section
        iterations_html = '<div class="reviews-list"><h3>Review Iterations</h3>'
        
        for iteration_num, version in enumerate(all_versions, 1):
            if not version.reviews:
                continue
            
            # Determine overall decision for this iteration
            all_accept = True
            try:
                for review_path in version.reviews:
                    with open(review_path, 'r') as f:
                        review_data = json.load(f)
                        decision = review_data.get('metadata', {}).get('decision', 'UNKNOWN')
                        if decision != 'ACCEPT':
                            all_accept = False
                            break
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read reviews for version {version.folder_name}: {e}[/yellow]")
                all_accept = False
            
            overall_decision = "ACCEPT" if all_accept else "REJECT"
            decision_class = "accept" if all_accept else "reject"
            
            iterations_html += f"""
            <div class="review-item">
                <a href="../iterations/{paper.get_base_paper_series()}_iteration_{iteration_num}.html">
                    Iteration {iteration_num}
                </a>
                <span class="decision {decision_class}">{overall_decision}</span>
            </div>
            """
        
        iterations_html += '</div>'
        
        # Generate links
        pdf_link = ""
        if paper.paper_pdf:
            pdf_filename = f"{paper.folder_name}.pdf"
            pdf_link = f'<a href="../pdfs/{pdf_filename}" class="btn">Download Final PDF</a>'
        
        # Convert abstract markdown to HTML
        abstract_html = markdown.markdown(paper.abstract)
        
        content = f"""
        <div class="paper-details">
            <h1>{paper.title}</h1>
            
            <div class="abstract">
                <h2 class="abstract-heading">Abstract</h2>
                {abstract_html}
            </div>
            
            <div class="links">
                {pdf_link}
                <a href="../index.html" class="btn secondary">← Back to Conference</a>
            </div>
            
            {iterations_html}
        </div>
        """
        
        html = self._wrap_html(
            title=paper.title,
            content=content,
            css_path="../css/style.css"
        )
        
        output_path = self.output_dir / 'papers' / f'{paper.folder_name}.html'
        output_path.write_text(html)
    
    def _copy_paper_pdf(self, paper: PaperFolder):
        """Copy paper PDF to output directory"""
        if paper.paper_pdf and paper.paper_pdf.exists():
            dest = self.output_dir / 'pdfs' / f'{paper.folder_name}.pdf'
            shutil.copy2(paper.paper_pdf, dest)
    
    def _generate_iteration_pages(self, final_paper: PaperFolder, all_versions: List[PaperFolder]):
        """Generate pages for each review iteration showing all reviews for that iteration"""
        base_series = final_paper.get_base_paper_series()
        
        for iteration_num, version in enumerate(all_versions, 1):
            if not version.reviews:
                continue
            
            console.print(f"[cyan]Generating iteration {iteration_num} page for: {final_paper.title}[/cyan]")
            
            # Determine overall decision
            all_accept = True
            try:
                for review_path in version.reviews:
                    with open(review_path, 'r') as f:
                        review_data = json.load(f)
                        decision = review_data.get('metadata', {}).get('decision', 'UNKNOWN')
                        if decision != 'ACCEPT':
                            all_accept = False
                            break
            except Exception:
                all_accept = False
            
            overall_decision = "ACCEPT" if all_accept else "REJECT"
            
            # Generate reviews HTML for this iteration
            reviews_html = f"""
            <h2>Iteration {iteration_num}: {overall_decision}</h2>
            <p class="iteration-info">This iteration contains {len(version.reviews)} review(s).</p>
            """
            
            # Add each review
            for review_idx, review_path in enumerate(version.reviews, 1):
                try:
                    with open(review_path, 'r') as f:
                        review_data = json.load(f)
                    
                    metadata = review_data.get('metadata', {})
                    review = review_data.get('review', {})
                    
                    decision = metadata.get('decision', 'UNKNOWN')
                    reviewed_at = metadata.get('reviewed_at', '')
                    
                    reviews_html += f"""
                    <div class="review-content">
                        <h3>Reviewer {review_idx}</h3>
                        
                        <p><strong>Decision:</strong> <span class="decision {'accept' if decision == 'ACCEPT' else 'reject'}">{decision}</span></p>
                        <p><strong>Reviewed:</strong> {reviewed_at}</p>
                        
                        <h4>Overall Assessment</h4>
                        <p>{review.get('overall_assessment', 'N/A')}</p>
                        
                        <h4>Strengths</h4>
                        <ul>
                        {''.join(f'<li>{s}</li>' for s in review.get('strengths', []))}
                        </ul>
                        
                        <h4>Weaknesses</h4>
                        <ul>
                        {''.join(f'<li>{w}</li>' for w in review.get('weaknesses', []))}
                        </ul>
                        
                        <h4>Detailed Comments</h4>
                        <p>{review.get('detailed_comments', 'N/A')}</p>
                    </div>
                    """
                    
                except Exception as e:
                    console.print(f"[red]Error processing review {review_path}: {e}[/red]")
            
            # Add link to the PDF that was reviewed
            pdf_link = ""
            if version.paper_pdf:
                pdf_filename = f"{version.folder_name}.pdf"
                pdf_link = f'<a href="../pdfs/{pdf_filename}" class="btn">Download Reviewed Paper (Iteration {iteration_num})</a>'
            
            # Wrap in complete page structure
            content = f"""
            <div class="iteration-details">
                <h1>{final_paper.title}</h1>
                {reviews_html}
                
                <div class="links">
                    {pdf_link}
                    <a href="../papers/{final_paper.folder_name}.html" class="btn secondary">← Back to Final Paper</a>
                </div>
            </div>
            """
            
            html = self._wrap_html(
                title=f"Iteration {iteration_num} - {final_paper.title}",
                content=content,
                css_path="../css/style.css"
            )
            
            output_path = self.output_dir / 'iterations' / f'{base_series}_iteration_{iteration_num}.html'
            output_path.write_text(html)
    
    def _generate_review_pages(self, paper: PaperFolder):
        """Generate individual review pages"""
        for i, review_path in enumerate(paper.reviews, 1):
            try:
                with open(review_path, 'r') as f:
                    review_data = json.load(f)
                
                metadata = review_data.get('metadata', {})
                review = review_data.get('review', {})
                
                decision = metadata.get('decision', 'UNKNOWN')
                reviewed_at = metadata.get('reviewed_at', '')
                
                # Add link to the PDF that was reviewed
                pdf_link = ""
                if paper.paper_pdf:
                    pdf_filename = f"{paper.folder_name}.pdf"
                    pdf_link = f'<a href="../pdfs/{pdf_filename}" class="btn">Download Reviewed Paper</a>'
                
                # Format review content
                review_html = f"""
                <div class="review-content">
                    <h1>Review #{i} for: {paper.title}</h1>
                    
                    <p><strong>Decision:</strong> <span class="decision">{decision}</span></p>
                    <p><strong>Reviewed:</strong> {reviewed_at}</p>
                    
                    <h2>Overall Assessment</h2>
                    <p>{review.get('overall_assessment', 'N/A')}</p>
                    
                    <h2>Strengths</h2>
                    <ul>
                    {''.join(f'<li>{s}</li>' for s in review.get('strengths', []))}
                    </ul>
                    
                    <h2>Weaknesses</h2>
                    <ul>
                    {''.join(f'<li>{w}</li>' for w in review.get('weaknesses', []))}
                    </ul>
                    
                    <h2>Detailed Comments</h2>
                    <p>{review.get('detailed_comments', 'N/A')}</p>
                    
                    <div class="links">
                        {pdf_link}
                        <a href="../papers/{paper.folder_name}.html" class="btn secondary">← Back to Final Paper</a>
                    </div>
                </div>
                """
                
                html = self._wrap_html(
                    title=f"Review #{i} - {paper.title}",
                    content=review_html,
                    css_path="../css/style.css"
                )
                
                review_id = review_path.stem
                output_path = self.output_dir / 'reviews' / f'{paper.folder_name}_{review_id}.html'
                output_path.write_text(html)
                
            except Exception as e:
                console.print(f"[red]Error generating review page for {review_path}: {e}[/red]")
    
    def _wrap_html(self, title: str, content: str, css_path: str = "css/style.css") -> str:
        """Wrap content in HTML template
        
        Args:
            title: Page title
            content: HTML content for the page
            css_path: Relative path to CSS file (default: 'css/style.css' for root pages,
                     use '../css/style.css' for pages in subdirectories)
        """
        
        # Include SVG as first child of body if provided
        svg_html = f"\n    {self.svg_content}" if self.svg_content else ""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{css_path}">
</head>
<body>{svg_html}
    <div class="container">
        <div class="content">
            {content}
        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>Generated by Buffy Bot Static Site Generator</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
    </footer>
</body>
</html>
"""


@click.command()
@click.option('--papers-dir', default='papers', help='Path to papers directory (default: papers)')
@click.option('--output-dir', default='conference_site', help='Output directory for static site (default: conference_site)')
@click.option('--landing-page', default='landing_page.md', help='Markdown file for landing page content (required, default: landing_page.md)')
@click.option('--tech-docs', default='tech_explanation.md', help='Markdown file for technical documentation (required, default: tech_explanation.md)')
@click.option('--custom-css', default=None, help='Optional custom CSS file to use instead of default styling')
@click.option('--svg-file', default=None, help='Optional SVG file to include as first child of body in each page')
def generate_site(papers_dir: str, output_dir: str, landing_page: str, tech_docs: str, custom_css: str, svg_file: str):
    """Generate static website from conference paper folders
    
    Both landing-page and tech-docs markdown files are required.
    The generator will fail if either file is missing.
    
    Optionally provide a custom CSS file with --custom-css to override the default styling.
    Optionally provide an SVG file with --svg-file to include as the first child of body in each page.
    """
    
    console.print("[bold cyan]Static Site Generator for Conference Papers[/bold cyan]\n")
    
    papers_path = Path(papers_dir)
    output_path = Path(output_dir)
    landing_path = Path(landing_page)
    tech_path = Path(tech_docs)
    custom_css_path = Path(custom_css) if custom_css else None
    svg_path = Path(svg_file) if svg_file else None
    
    if not papers_path.exists():
        console.print(f"[red]Error: Papers directory '{papers_dir}' does not exist[/red]")
        return
    
    # Validate SVG file if provided
    if svg_path and not svg_path.exists():
        console.print(f"[red]Error: SVG file not found: {svg_path}[/red]")
        return
    
    # Create generator
    generator = StaticSiteGenerator(papers_path, output_path, svg_path)
    
    # Scan papers
    generator.scan_papers()
    
    # Generate site (will fail if markdown files are missing)
    try:
        generator.generate_site(landing_path, tech_path, custom_css_path)
        console.print("\n[green]✓ Static site generated successfully![/green]")
        console.print(f"[blue]Open {output_path / 'index.html'} in your browser to view the site[/blue]")
    except FileNotFoundError as e:
        console.print(f"\n[red]✗ Failed to generate site: {e}[/red]")
        return


if __name__ == '__main__':
    generate_site()

