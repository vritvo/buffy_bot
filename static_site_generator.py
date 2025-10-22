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
        """Find the paper PDF file"""
        matches = list(self.path.glob('paper*.pdf')) + list(self.path.glob('*.pdf'))
        return matches[0] if matches else None
    
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
    
    def __init__(self, papers_dir: Path, output_dir: Path):
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.papers: List[PaperFolder] = []
        
    def scan_papers(self):
        """Scan papers directory and identify final accepted versions"""
        console.print("[cyan]Scanning papers directory...[/cyan]")
        
        all_folders = [f for f in self.papers_dir.iterdir() if f.is_dir()]
        all_papers = [PaperFolder(f) for f in all_folders]
        
        # Group by base paper series
        paper_series: Dict[str, List[PaperFolder]] = {}
        for paper in all_papers:
            series = paper.get_base_paper_series()
            if series not in paper_series:
                paper_series[series] = []
            paper_series[series].append(paper)
        
        # For each series, find the highest version that's accepted
        for series, versions in paper_series.items():
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
    
    def generate_site(self, landing_md: Optional[Path], tech_md: Optional[Path]):
        """Generate complete static site"""
        console.print("\n[cyan]Generating static site...[/cyan]")
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'papers').mkdir(exist_ok=True)
        (self.output_dir / 'reviews').mkdir(exist_ok=True)
        (self.output_dir / 'pdfs').mkdir(exist_ok=True)
        (self.output_dir / 'css').mkdir(exist_ok=True)
        
        # Generate CSS
        self._generate_css()
        
        # Generate landing page
        self._generate_landing_page(landing_md)
        
        # Generate tech documentation
        if tech_md and tech_md.exists():
            self._generate_tech_page(tech_md)
        
        # Generate paper pages
        for paper in self.papers:
            self._generate_paper_page(paper)
            self._copy_paper_pdf(paper)
            self._generate_review_pages(paper)
        
        console.print(f"\n[green]✓ Site generated successfully in {self.output_dir}[/green]")
    
    def _generate_css(self):
        """Generate CSS stylesheet"""
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
    background: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 40px 0;
    margin-bottom: 40px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

header h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
}

header p {
    font-size: 1.2em;
    opacity: 0.9;
}

nav {
    background: white;
    padding: 15px 0;
    margin-bottom: 30px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    position: sticky;
    top: 0;
    z-index: 100;
}

nav ul {
    list-style: none;
    display: flex;
    gap: 30px;
    flex-wrap: wrap;
}

nav a {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}

nav a:hover {
    color: #764ba2;
}

.content {
    background: white;
    padding: 40px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 30px;
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
    gap: 15px;
    flex-wrap: wrap;
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

.review-content {
    background: #f8f9fa;
    padding: 30px;
    border-radius: 8px;
    margin: 20px 0;
}

.review-content h4 {
    color: #667eea;
    margin-top: 20px;
    margin-bottom: 10px;
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

.markdown-content h1,
.markdown-content h2,
.markdown-content h3 {
    color: #667eea;
    margin-top: 30px;
    margin-bottom: 15px;
}

.markdown-content h1 {
    font-size: 2em;
    border-bottom: 2px solid #667eea;
    padding-bottom: 10px;
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

@media (max-width: 768px) {
    .paper-grid {
        grid-template-columns: 1fr;
    }
    
    header h1 {
        font-size: 2em;
    }
    
    .content {
        padding: 20px;
    }
    
    nav ul {
        gap: 15px;
    }
}
"""
        css_path = self.output_dir / 'css' / 'style.css'
        css_path.write_text(css)
    
    def _generate_landing_page(self, landing_md: Optional[Path]):
        """Generate landing page"""
        console.print("[cyan]Generating landing page...[/cyan]")
        
        # Read landing page content
        landing_content = ""
        if landing_md and landing_md.exists():
            landing_content = markdown.markdown(landing_md.read_text())
        else:
            landing_content = """
            <h2>Welcome to the Conference Papers Archive</h2>
            <p>This site contains the final accepted papers from our academic conference,
            complete with peer reviews and supporting materials.</p>
            """
        
        # Generate paper list
        papers_html = '<div class="paper-grid">'
        for paper in self.papers:
            abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            papers_html += f"""
            <div class="paper-card">
                <h3>{paper.title}</h3>
                <div class="abstract">{abstract_preview}</div>
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
            
            <h2 style="margin-top: 40px; color: #667eea;">Accepted Papers</h2>
            {papers_html}
            """,
            active_page="home"
        )
        
        (self.output_dir / 'index.html').write_text(html)
    
    def _generate_tech_page(self, tech_md: Path):
        """Generate technical documentation page"""
        console.print("[cyan]Generating technical documentation...[/cyan]")
        
        content = markdown.markdown(tech_md.read_text(), extensions=['fenced_code', 'tables'])
        
        html = self._wrap_html(
            title="Technical Documentation",
            content=f'<div class="markdown-content">{content}</div>',
            active_page="tech"
        )
        
        (self.output_dir / 'technical_documentation.html').write_text(html)
    
    def _generate_paper_page(self, paper: PaperFolder):
        """Generate individual paper page"""
        console.print(f"[cyan]Generating page for: {paper.title}[/cyan]")
        
        # Generate reviews section
        reviews_html = '<div class="reviews-list"><h3>Peer Reviews</h3>'
        for i, review_path in enumerate(paper.reviews, 1):
            try:
                with open(review_path, 'r') as f:
                    review_data = json.load(f)
                    decision = review_data.get('metadata', {}).get('decision', 'UNKNOWN')
                    review_id = review_path.stem
                    reviews_html += f"""
                    <div class="review-item">
                        <a href="../reviews/{paper.folder_name}_{review_id}.html">
                            Review #{i}
                        </a>
                        <span class="decision">{decision}</span>
                    </div>
                    """
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read review {review_path}: {e}[/yellow]")
        
        reviews_html += '</div>'
        
        # Generate links
        pdf_link = ""
        if paper.paper_pdf:
            pdf_filename = f"{paper.folder_name}.pdf"
            pdf_link = f'<a href="../pdfs/{pdf_filename}" class="btn">Download PDF</a>'
        
        content = f"""
        <div class="paper-details">
            <h2>{paper.title}</h2>
            
            <div class="abstract">
                <h3 style="color: #667eea; margin-bottom: 15px;">Abstract</h3>
                {paper.abstract}
            </div>
            
            <div class="links">
                {pdf_link}
            </div>
            
            {reviews_html}
        </div>
        """
        
        html = self._wrap_html(
            title=paper.title,
            content=content,
            active_page=""
        )
        
        output_path = self.output_dir / 'papers' / f'{paper.folder_name}.html'
        output_path.write_text(html)
    
    def _copy_paper_pdf(self, paper: PaperFolder):
        """Copy paper PDF to output directory"""
        if paper.paper_pdf and paper.paper_pdf.exists():
            dest = self.output_dir / 'pdfs' / f'{paper.folder_name}.pdf'
            shutil.copy2(paper.paper_pdf, dest)
    
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
                    pdf_link = f'<a href="../pdfs/{pdf_filename}" class="btn">Download Reviewed PDF</a>'
                
                # Format review content
                review_html = f"""
                <div class="review-content">
                    <h2>Review #{i} for: {paper.title}</h2>
                    
                    <p><strong>Decision:</strong> <span class="decision">{decision}</span></p>
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
                    
                    <div class="links">
                        {pdf_link}
                        <a href="../papers/{paper.folder_name}.html" class="btn secondary">← Back to Paper</a>
                    </div>
                </div>
                """
                
                html = self._wrap_html(
                    title=f"Review #{i} - {paper.title}",
                    content=review_html,
                    active_page=""
                )
                
                review_id = review_path.stem
                output_path = self.output_dir / 'reviews' / f'{paper.folder_name}_{review_id}.html'
                output_path.write_text(html)
                
            except Exception as e:
                console.print(f"[red]Error generating review page for {review_path}: {e}[/red]")
    
    def _wrap_html(self, title: str, content: str, active_page: str = "") -> str:
        """Wrap content in HTML template"""
        
        # Build navigation
        nav_items = [
            ('index.html', 'Home', 'home'),
            ('technical_documentation.html', 'Technical Documentation', 'tech'),
        ]
        
        nav_html = '<ul>'
        for href, label, page_id in nav_items:
            active = 'style="color: #764ba2;"' if page_id == active_page else ''
            nav_html += f'<li><a href="{href}" {active}>{label}</a></li>'
        nav_html += '</ul>'
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>Conference Papers Archive</h1>
            <p>Academic papers with peer reviews</p>
        </div>
    </header>
    
    <nav>
        <div class="container">
            {nav_html}
        </div>
    </nav>
    
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
@click.option('--landing-page', default='landing_page.md', help='Markdown file for landing page content (default: landing_page.md)')
@click.option('--tech-docs', default='tech_explanation.md', help='Markdown file for technical documentation (default: tech_explanation.md)')
def generate_site(papers_dir: str, output_dir: str, landing_page: str, tech_docs: str):
    """Generate static website from conference paper folders"""
    
    console.print("[bold cyan]Static Site Generator for Conference Papers[/bold cyan]\n")
    
    papers_path = Path(papers_dir)
    output_path = Path(output_dir)
    landing_path = Path(landing_page) if landing_page else None
    tech_path = Path(tech_docs) if tech_docs else None
    
    if not papers_path.exists():
        console.print(f"[red]Error: Papers directory '{papers_dir}' does not exist[/red]")
        return
    
    # Create generator
    generator = StaticSiteGenerator(papers_path, output_path)
    
    # Scan papers
    generator.scan_papers()
    
    # Generate site
    generator.generate_site(landing_path, tech_path)
    
    console.print(f"\n[green]✓ Static site generated successfully![/green]")
    console.print(f"[blue]Open {output_path / 'index.html'} in your browser to view the site[/blue]")


if __name__ == '__main__':
    generate_site()

