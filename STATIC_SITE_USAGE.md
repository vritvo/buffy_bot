# Static Site Generator - Usage Guide

## Overview

The static site generator creates a beautiful, responsive website from your conference paper folders. It automatically identifies the final accepted version of each paper and generates a complete website with landing page, technical documentation, paper pages, and review pages.

## Quick Start

```bash
# Generate site with default settings (requires landing_page.md and tech_explanation.md)
uv run python static_site_generator.py

# Open the generated site in your browser
open conference_site/index.html
```

**Note:** Both `landing_page.md` and `tech_explanation.md` are **required**. The generator will fail with a clear error message if either file is missing.

## Command Options

```bash
uv run python static_site_generator.py \
  --papers-dir papers \
  --output-dir conference_site \
  --landing-page landing_page.md \
  --tech-docs tech_explanation.md
```

### Options

- `--papers-dir`: Directory containing paper folders from `run-conference` (default: `papers`)
- `--output-dir`: Where to generate the static site (default: `conference_site`)
- `--landing-page`: Markdown file for landing page content (required, default: `landing_page.md`)
- `--tech-docs`: Markdown file for technical documentation (required, default: `tech_explanation.md`)
- `--custom-css`: Optional custom CSS file to use instead of default styling

Both markdown files must exist or the generator will fail with an error.

## How It Works

### 1. Paper Detection

The generator scans the papers directory and:
- Groups papers by base name (e.g., all versions of "Spikes_Journey_...")
- Identifies version numbers (e.g., `_v2`, `_v2_v2`)
- Selects the highest version with all reviews marked as ACCEPT
- Falls back to the latest version if no accepted version exists

### 2. Content Extraction

For each paper, the generator:
- Extracts title from YAML frontmatter or first heading
- Extracts abstract from the Abstract section
- Finds the paper PDF
- Collects all review JSON files

### 3. Site Generation

The generator creates:

**Landing Page (`index.html`)**
- Displays content from your landing page markdown
- Shows grid of all accepted papers with abstracts
- Links to individual paper pages

**Technical Documentation (`technical_documentation.html`)**
- Renders your tech documentation markdown as HTML
- Includes syntax highlighting for code blocks
- Displays conference run GIF at the bottom if available

**Paper Pages (`papers/*.html`)**
- Full paper abstract
- Download link for the final PDF
- List of all peer reviews with links

**Review Pages (`reviews/*.html`)**
- Complete review content (decision, strengths, weaknesses, detailed comments)
- Link to download the PDF that was reviewed
- Link back to the paper page

**Assets**
- Responsive CSS with modern design
- Copied paper PDFs for download

## Customizing Content

### Landing Page

Edit `landing_page.md` to customize your landing page. Use standard Markdown:

```markdown
# Welcome to My Conference

This archive contains papers from our conference on...

## About

Our conference focuses on...
```

### Technical Documentation

Edit `tech_explanation.md` to document your system. Supports:
- Headers, lists, and text formatting
- Code blocks with syntax highlighting
- Tables

### Custom Styling

You can provide your own CSS file to completely customize the site's appearance:

```bash
uv run python static_site_generator.py --custom-css my_style.css
```

Your custom CSS file should define all the necessary classes and styles. Key classes to include:
- `.container`, `.content`, `.paper-card`, `.paper-grid`
- `.btn`, `.btn.secondary`
- `.reviews-list`, `.review-item`, `.review-content`
- `.markdown-content`, `.abstract`, `.abstract-heading`
- `header`, `nav`, `footer`

The generator will copy your CSS file to `conference_site/css/style.css` and all pages will use it.

### Adding a Conference Run GIF

You can add an animated GIF showing a conference run to the Technical Documentation page:

1. **Create or convert your recording to GIF:**
   ```bash
   # If you have a .cast file from asciinema, you can convert it using agg or similar tools
   # Or use any screen recording converted to GIF
   ```

2. **Place the GIF:**
   ```
   docs/gifs/conference_run.gif   # Or gifs/conference_run.gif at project root
   ```

3. **Generate the site:**
   The static site generator will automatically:
   - Copy the `gifs/` directory to the output
   - Display `conference_run.gif` at the bottom of the Technical Documentation page
   - Style it with a subtle border and shadow

The GIF will appear centered below your technical documentation content with responsive sizing.

## Output Structure

```
conference_site/
├── index.html                  # Landing page
├── technical_documentation.html # Tech docs
├── css/
│   └── style.css              # Responsive styles
├── gifs/                       # Animated GIFs (if present)
│   └── conference_run.gif
├── papers/                     # One page per paper
│   ├── 20251021_014943_Spikes_Journey_*.html
│   └── ...
├── iterations/                 # One page per iteration
│   ├── *_iteration_1.html
│   └── ...
├── pdfs/                       # Paper PDFs
│   ├── 20251021_014943_Spikes_Journey_*.pdf
│   └── ...
└── fonts/                      # Custom fonts (if present)
    └── ...
```

## Design Features

The generated site includes:

✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Modern UI** - Clean gradient headers and card-based layouts
✅ **Easy Navigation** - Sticky nav bar and consistent breadcrumbs
✅ **Readable Typography** - Optimized fonts and spacing
✅ **Professional Styling** - Subtle shadows and smooth transitions
✅ **Clean CSS** - All styling in one CSS file, no inline styles

## Deploying Your Site

The generated site is 100% static HTML/CSS. Deploy it anywhere:

### GitHub Pages

```bash
cd conference_site
git init
git add .
git commit -m "Initial site"
git remote add origin https://github.com/yourusername/yourrepo.git
git push -u origin main
```

Then enable GitHub Pages in your repo settings.

### Netlify

1. Drag and drop the `conference_site` folder to Netlify
2. Site is live!

### Any Web Server

Copy the `conference_site` folder to your web server's public directory.

## Troubleshooting

### Papers not appearing

- Check that paper folders contain a `.md` file and `.pdf` file
- Verify reviews are in a `reviews/` subdirectory
- Look for warnings in the generator output

### Missing abstracts

- Ensure papers have an `## Abstract` section
- Check that abstracts come after YAML frontmatter (if present)

### Missing markdown files

- The generator requires both `landing_page.md` and `tech_explanation.md`
- Create these files or specify different paths with `--landing-page` and `--tech-docs`
- The generator will fail immediately if either file is missing

### Broken links

- Run the generator from the project root directory
- Check that `--papers-dir` points to the correct location

## Advanced Usage

### Filtering Papers

To generate a site for only specific papers, temporarily move unwanted paper folders out of the papers directory before running the generator.

### Post-Generation Styling

After generation, you can edit `conference_site/css/style.css` directly to tweak colors, fonts, and layout:

```css
/* Primary gradient colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent colors */
color: #667eea;
border-left: 4px solid #667eea;
```

Note: Changes to the generated CSS will be overwritten on the next generation. Use `--custom-css` to preserve your styling across regenerations.

### Multiple Conferences

Generate separate sites for different conferences:

```bash
# Conference 1
uv run python static_site_generator.py \
  --papers-dir papers/conference1 \
  --output-dir sites/conference1

# Conference 2
uv run python static_site_generator.py \
  --papers-dir papers/conference2 \
  --output-dir sites/conference2
```

## Support

For issues or questions:
1. Check the main README.md
2. Review the source code in `static_site_generator.py`
3. Examine the generated HTML for debugging

