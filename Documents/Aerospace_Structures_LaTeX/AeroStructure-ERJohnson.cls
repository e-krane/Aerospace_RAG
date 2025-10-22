\newcommand\classname{AeroStructure-ERJohnson}
\newcommand\lastmodifieddate{2021/12/24}
\newcommand\versionnumber{0.1}

% Are we printing crop marks?
\newif\if@cropmarkson \@cropmarksontrue
\newif\ifAppendix\Appendixfalse%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{\classname}[\lastmodifieddate\space\versionnumber]

\setlength{\paperheight}{11truein}%
\setlength{\paperwidth}{8.5truein}%

\special{papersize[8.5in,11in}%

\DeclareOption{draft}{\PassOptionsToPackage{draft}{graphicx}}
\DeclareOption{a4paper}{\PassOptionsToPackage{a4}{crop}}
\DeclareOption{centre}{\PassOptionsToPackage{center}{crop}}
\DeclareOption{crop}{\PassOptionsToPackage{width=9.5in,height=12in,cam}{crop}\global\@cropmarksontrue}
\DeclareOption{nocrop}{\PassOptionsToPackage{width=8.5in,height=11in,off}{crop}\global\@cropmarksonfalse}
\DeclareOption{info}{\PassOptionsToPackage{info}{crop}}
\DeclareOption{noinfo}{\PassOptionsToPackage{noinfo}{crop}}

%\ExecuteOptions{a4paper,crop,centre,info}
\ExecuteOptions{nocrop,noinfo}

\ProcessOptions

\renewcommand\normalsize{%
   \@setfontsize\normalsize{10}{12}%
   \abovedisplayskip 12\p@ \@plus2\p@ \@minus5\p@%
   \abovedisplayshortskip \z@ \@plus3\p@%
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@%
   \belowdisplayskip \abovedisplayskip%
   \let\@listi\@listI}

% Load all necessary packages
\RequirePackage{ifxetex,ifluatex}%
\usepackage[no-math]{fontspec}%
\usepackage{amsthm}
\usepackage[]{newtxmath} % 
\usepackage{mathtools}
\usepackage{mathspec}
\defaultfontfeatures{Ligatures=TeX}%
%\setmainfont[Ligatures=TeX,Path = ./ ,
%UprightFont= TimesNewRomPSPro,
%BoldFont= TimesNewRomPSPro-Bol,
%ItalicFont=TimesNewRomPSPro-Ita,
%BoldItalicFont=TimesNewRomPSPro-BolIta,
%]{TimesNewRomPSPro}%
\setmainfont{TeX Gyre Termes}
%\RequirePackage[defaultmathsizes,italic]{mathastext}%
%\setsansfont[Ligatures=TeX, Path = ./ ,
%UprightFont= ArialMTStd ,
%BoldFont=ArialMTStd-Bold ,
%ItalicFont=ArialMTStd-Italic ,
%BoldItalicFont=ArialMTStd-BoldItalic,
%]{ArialMTStd}
\setsansfont{texgyreheros}[
%        Scale=MatchLowercase,% or MatchUppercase
        Extension=.otf,
        UprightFont=*-regular,
        ItalicFont=*-italic,
        BoldFont=*-bold,
        BoldItalicFont=*-bolditalic,
        ]
\RequirePackage{crop,graphicx,amsfonts,makeidx,array,soul}%
\RequirePackage{framed,stfloats,float,courier,ragged2e}%
\RequirePackage[figuresright]{rotating}%
\RequirePackage[export]{adjustbox}%
\RequirePackage{ifoddpage}%

%%Sansbold definitions
\newcommand{\sansbold}[1]{\textbf{\textsf{#1}}}%
\newcommand{\eqsansbold}[1]{\textbf{{#1}}}%


% Not sure if needed.
\newcommand\@ptsize{0}

% Set twoside printing
\@twosidetrue

% Marginal notes are on the outside edge
\@mparswitchtrue

% Default font sizes
\def\aboveskip{%
\abovedisplayskip=0pt%
\abovedisplayshortskip=0pt%
}%

\renewcommand\normalsize{%
   \@setfontsize\normalsize{10}{12}%
   \abovedisplayskip 12\p@ \@plus2\p@ \@minus5\p@%
   \abovedisplayshortskip \z@ \@plus3\p@%
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@%
   \belowdisplayskip \abovedisplayskip%
   \let\@listi\@listI}%
\normalsize%
\let\@bls\baselineskip%

\newcommand\small{%
   \@setfontsize\small{9}{11}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\newcommand\footnotesize{%
   \@setfontsize\footnotesize{8}{9}%
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
%\newcommand\little{\@setfontsize\tiny\@ixpt\@ixpt}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{18}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}

% Line spacing
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}

% Paragraph dimensions and inter-para spacing
\setlength\parskip{0\p@}
\setlength\parindent{1.5em}

% Set inter-para skips
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}

% Page break penalties
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301

% Disallow widows and orphans
\clubpenalty 10000
\widowpenalty 10000

% Disable page breaks before equations, allow pagebreaks after
% equations and discourage widow lines before equations.
\displaywidowpenalty 100
\predisplaypenalty   10000
\postdisplaypenalty  0

% Allow breaking the page in the middle of a paragraph
\interlinepenalty 0

% Disallow breaking the page after a hyphenated line
\brokenpenalty 10000

% Hyphenation; don't split words into less than three characters
\lefthyphenmin=3
\righthyphenmin=3

%
% Set page layout dimensions
%
\setlength\headheight{8\p@}         % height of running head
\setlength\topmargin{58\p@}         % head margin
\addtolength\topmargin{-1in}        % subtract out the 1 inch driver margin

\setlength\topskip{10\p@}           % height of first line of text
\AtBeginDocument{\setlength{\topskip}{\fontcharht\font`T}}%
\setlength\headsep{63.75\p@}           % space below running head --
\addtolength\headsep{-\topskip}     %   base to base with first line of text

\setlength\footskip{\z@}            % space above footer line
\setlength\maxdepth{.5\topskip}     % pages can be short or deep by half a line?

\setlength\textwidth{37.5pc}          % text measure excluding margins

\setlength\textheight{49\baselineskip} % 50 lines on a full page, extracted from supplied FrameMaker files
\addtolength\textheight{\topskip}      %    including the first
                                       %    line on the page

% Set the margins
% Margin paras are not required. Set lengths to zero.
\setlength\marginparsep{18\p@}
\setlength\marginparpush{6\p@}
\setlength\marginparwidth{40\p@}

\setlength\oddsidemargin{7.5pc}
\addtolength\oddsidemargin{-1in}    % subtract out the 1 inch driver margin
\setlength\@tempdima{\paperwidth}
\addtolength\@tempdima{-\textwidth}
\addtolength\@tempdima{-7.5pc}
\setlength\evensidemargin{\@tempdima}
\addtolength\evensidemargin{-1in}

\setlength\columnsep{12\p@}          % space between columns for double-column text
\setlength\columnseprule{0\p@}      % width of rule between two columns

% Footnotes
\setlength\footnotesep{8\p@}     % space between footnotes
% space between text and footnote
\setlength{\skip\footins}{12\p@ plus3\p@}%

% Float placement parameters

% The total number of floats that can be allowed on a page.
\setcounter{totalnumber}{10}
% The maximum number of floats at the top and bottom of a page.
\setcounter{topnumber}{5}
\setcounter{bottomnumber}{5}
% The maximum part of the top or bottom of a text page that can be
% occupied by floats. This is set so that at least four lines of text
% fit on the page.
\renewcommand\topfraction{.921}
\renewcommand\bottomfraction{.921}
% The minimum amount of a text page that must be occupied by text.
% This should accomodate four lines of text.
\renewcommand\textfraction{.079}
% The minimum amount of a float page that must be occupied by floats.
\renewcommand\floatpagefraction{.887}

% The same parameters repeated for double column output
\renewcommand\dbltopfraction{.88}
\renewcommand\dblfloatpagefraction{.88}

% Space between floats
\setlength\floatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
% Space between floats and text
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
% Space above and below an inline figure
\setlength\intextsep   {18\p@ \@plus 2\p@ \@minus 2\p@}

% For double column floats
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}

% Space left at top, bottom and inbetween floats on a float page.
\setlength\@fptop{0\p@}         % no space above float page figures
\setlength\@fpsep{12\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}

% The same for double column
\setlength\@dblfptop{0\p@}
\setlength\@dblfpsep{12\p@ \@plus 1fil}
\setlength\@dblfpbot{0\p@ \@plus 2fil}

% Page styles
\sodef\logofltrack{}{1.25\p@}{4\p@}{0pt}%
\sodef\logotrack{}{1\p@}{3\p@}{0pt}%

\newcommand{\jobtitle}{Aerospace Structures}%

  \def\ps@headings{%
\def\@oddfoot{\vbox{%
\vspace{9.75\p@}\noindent\rule{\textwidth}{0.5\p@}\par%
\vspace{1.5\p@}%
{\fontsize{10}{10}\fontshape{it}\selectfont\jobtitle}\hfill{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}%
}%
\smash{\begin{picture}(0,0)(0,0)\put(88,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedright%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
}%
\def\@evenfoot{\smash{\begin{picture}(0,0)(0,0)\put(-150,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedleft%
-1\raisebox{-2.5\p@}{---}\\0\raisebox{-2.5\p@}{---}\\+1\raisebox{-2.5\p@}{---}}}\end{picture}}%
\vbox{\vspace{9.75\p@}\noindent\rule{\textwidth}{0.5\p@}\par%
\vspace{1.5\p@}%
{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}\hfill{\fontsize{10}{10}\fontshape{it}\selectfont\jobtitle}%
}%
}%
      \def\@evenhead{\vbox{\noindent\rule{\textwidth}{2\p@}\par%
\vspace{2\p@}%
\fontsize{8}{8}\selectfont%
\noindent{\if@mainmatter{\textsf{\textbf{Article~\thesection}}}\else\relax\fi}{\relax}\hfill\par%
\vspace{7\p@}%
\noindent\rule{\textwidth}{0.5\p@}%
}%
}%
      \def\@oddhead{\vbox{\noindent\rule{\textwidth}{2\p@}\par%
\vspace{2\p@}%
\fontsize{8}{8}\selectfont%
\noindent{\if@mainmatter{\hfill\textsf{\textbf{\vphantom{y}\rightmark}}}\else\relax\fi}{\relax}\par%
\vspace{7\p@}%
\noindent\rule{\textwidth}{0.5\p@}%
}%
}%%
      \let\@mkboth\markboth
    \def\chaptermark##1{\markboth{##1}{##1}}%
    \def\sectionmark##1{%
      \markright {{%
        \ifnum \c@secnumdepth >\z@
%          \thesection\ignorespaces\enskip\ignorespaces%
        \fi
        ##1}}}}

\def\ps@plain{%
\def\@oddfoot{\smash{\begin{picture}(0,0)(0,0)\put(539.5,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedright%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
\vbox{\vspace{9.75\p@}\noindent\rule{\textwidth}{0.5\p@}\par%
\vspace{1.5\p@}%
{\fontsize{9.5}{9.5}\selectfont\textit{\jobtitle\ (c)}\ Eric Raymond Johnson. CC BY NC SA \underline{\url{https://doi.org/10.21061/AerospaceStructures}}}\hfill{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}%
}%
}%
 \let\@evenfoot\relax%
%  \def\@oddfoot{\hfill{{\fontsize{10}{12}\selectfont\thepage}}}%
  \let\@evenfoot\@oddfoot%
  \let\@oddhead\relax \let\@evenhead\relax}%

\def\ps@fmplain{%
\def\@oddfoot{\smash{\begin{picture}(0,0)(0,0)\put(539.5,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedright%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
\vbox{\vspace{9.75\p@}\noindent\rule{\textwidth}{0\p@}\par%
\vspace{1.5\p@}%
{\fontsize{9.5}{9.5}\selectfont\phantom{\textit{\jobtitle\ (\textcopyright)\ Eric Raymond Johnson. CC By NC SA \underline{\url{https://doi.org/10.21061/AerospaceStructures}}}}\hfill{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}%
}%
}}%
 \let\@evenfoot\relax%
  \let\@oddhead\relax \let\@evenhead\relax}%

  \def\ps@fmheadings{%
\def\@oddfoot{\smash{\begin{picture}(0,0)(0,0)\put(539.5,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedright%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
\vbox{\vspace{9.75\p@}\noindent\rule{\textwidth}{0\p@}\par%
\vspace{1.5\p@}%
{\fontsize{10}{10}\fontshape{it}\selectfont\phantom{\jobtitle}}\hfill{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}%
}%
}%
\def\@evenfoot{\vbox{\smash{\begin{picture}(0,0)(0,0)\put(-148,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedleft%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
\vspace{9.75\p@}\noindent\rule{\textwidth}{0\p@}\par%
\vspace{1.5\p@}%
{\fontsize{12}{12}\fontseries{b}\selectfont\thepage}\hfill{\fontsize{10}{10}\fontshape{it}\selectfont\phantom{\jobtitle}}%
}%
}%
      \def\@evenhead{\relax}%\vbox{\noindent\rule{\textwidth}{2\p@}\par%
%\vspace{2\p@}%
%\fontsize{8}{8}\selectfont%
%\noindent{\if@mainmatter{\textsf{\textbf{Article~\thesection}}}\else\relax\fi}{\relax}\hfill\par%
%\vspace{7\p@}%
%\noindent\rule{\textwidth}{0.5\p@}%
%}%
%}%
      \def\@oddhead{\relax}%\vbox{\noindent\rule{\textwidth}{2\p@}\par%
%\vspace{2\p@}%
%\fontsize{8}{8}\selectfont%
%\noindent{\if@mainmatter{\hfill\textsf{\textbf{\vphantom{y}\leftmark}}}\else\relax\fi}{\relax}\par%
%\vspace{7\p@}%
%\noindent\rule{\textwidth}{0.5\p@}%
%}%
%}%%
      \let\@mkboth\markboth
    \def\chaptermark##1{\markboth{##1}{##1}}%
    \def\sectionmark##1{%
      \markright {{%
        \ifnum \c@secnumdepth >\z@
%          \thesection\ignorespaces\enskip\ignorespaces%
        \fi
        ##1}}}}

  \def\ps@empty{%
\def\@oddfoot{\vbox{%
\vspace{9.75\p@}\noindent\rule{0pt}{0.5\p@}\par%
\vspace{1.5\p@}%
{\fontsize{10}{10}\fontshape{it}\selectfont\phantom{\jobtitle}}\hfill{\fontsize{12}{12}\fontseries{b}\selectfont\phantom{\thepage}}%
}%
\smash{\begin{picture}(0,0)(0,0)\put(88,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedright%
\raisebox{-2.5\p@}{---}-1\\\raisebox{-2.5\p@}{---}0\\\raisebox{-2.5\p@}{---}+1}}\end{picture}}%
}%
\def\@evenfoot{\smash{\begin{picture}(0,0)(0,0)\put(-150,35){%
\parbox[t]{5pc}{\baselineskip12\p@\normalfont\raggedleft%
-1\raisebox{-2.5\p@}{---}\\0\raisebox{-2.5\p@}{---}\\+1\raisebox{-2.5\p@}{---}}}\end{picture}}%
\vbox{\vspace{9.75\p@}\noindent\rule{0pt}{0.5\p@}\par%
\vspace{1.5\p@}%
{\fontsize{12}{12}\fontseries{b}\selectfont\phantom{\thepage}}\hfill{\fontsize{10}{10}\fontshape{it}\selectfont\phantom{\jobtitle}}%
}%
}%
      \def\@evenhead{\relax}%\vbox{\noindent\rule{\textwidth}{2\p@}\par%
%\vspace{2\p@}%
%\fontsize{8}{8}\selectfont%
%\noindent{\if@mainmatter{\textsf{\textbf{Article~\thesection}}}\else\relax\fi}{\relax}\hfill\par%
%\vspace{7\p@}%
%\noindent\rule{\textwidth}{0.5\p@}%
%}%
%}%
      \def\@oddhead{\relax}%\vbox{\noindent\rule{\textwidth}{2\p@}\par%
%\vspace{2\p@}%
%\fontsize{8}{8}\selectfont%
%\noindent{\if@mainmatter{\hfill\textsf{\textbf{\vphantom{y}\rightmark}}}\else\relax\fi}{\relax}\par%
%\vspace{7\p@}%
%\noindent\rule{\textwidth}{0.5\p@}%
%}%
%}%%
      \let\@mkboth\markboth
    \def\chaptermark##1{\markboth{##1}{##1}}%
    \def\sectionmark##1{%
      \markright {{%
        \ifnum \c@secnumdepth >\z@
%          \thesection\ignorespaces\enskip\ignorespaces%
        \fi
        ##1}}}}
        
        
%
% Sectional units
%

% Lowest level heading that takes a number by default
\setcounter{secnumdepth}{2}%

% Counters
\newcounter{part}
\newcounter{chapter}
\newcounter{section}[chapter]
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\newcounter{figure}[chapter]
\newcounter{table}[chapter]


% Form of the numbers
\newcommand\thepage{\arabic{page}}
\renewcommand\thepart{\Roman{part}}
\renewcommand\thechapter{\arabic{chapter}}
\renewcommand\thesection{\thechapter.\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}%
\renewcommand\theparagraph{\thesubsubsection.\arabic{paragraph}}
\renewcommand\thesubparagraph{\theparagraph.\arabic{subparagraph}}
\renewcommand\theequation{\thechapter.\arabic{equation}}
\newcommand\@chapapp{\chaptername}

\@addtoreset{equation}{chapter}%

% Form of the words
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}%
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\newcommand\refname{References}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.}
\newcommand\tablename{Table}

% Clearemptydoublepage should really clear the running heads too
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}

% Frontmatter, mainmatter and backmatter

\newif\if@mainmatter \@mainmattertrue

\newcommand\frontmatter{%
  \clearpage
  \@mainmatterfalse
  \pagenumbering{roman}}

\newcommand{\HalfTitle}[1]{\thispagestyle{empty}\bgroup\parindent\z@\raggedleft%
\null\vspace{142\p@}%
\fontsize{24}{28}\fontseries{b}\selectfont%
#1\par%
\egroup\clearpage}%

\newenvironment{seriespage}{\thispagestyle{empty}\bgroup\parindent\z@\raggedright%
%\null\vspace{10\p@}%
\null\vspace{-14\p@}%
\fontsize{12}{14}\selectfont\setlength{\parskip}{1\baselineskip}}{\par\egroup\clearpage}%

\newcommand{\Title}[1]{\thispagestyle{empty}\bgroup\parindent\z@\centering%
\null\vspace{114\p@}%
\fontsize{24}{28}\fontseries{b}\selectfont#1\vphantom{y}\par%
\egroup}%

\newcommand{\BookAuthor}[1]{\bgroup\parindent\z@\centering%
\vspace{7\p@}%
\fontsize{14}{16}\fontseries{b}\selectfont#1\vphantom{y}\par%
\egroup}

\newcommand{\BookDOI}[1]{\bgroup\parindent\z@\centering%
\vspace{112\p@}%
\normalfont\normalsize%
#1\vphantom{y}\par%
\egroup}

\newcommand{\BookEditor}[1]{\bgroup\parindent\z@\centering%
\normalfont\normalsize%
\vspace{168\p@}%
%\vfill%
#1\vphantom{y}\par%
\vspace{12.5\p@}%
\egroup}

\newbox\logobox%
\newdimen\logodimen%

\newcommand{\logo}{\bgroup\parindent\z@%
\setbox\logobox=\hbox{\includegraphics[width=102pt,height=64pt]{VTPublishing-Logo-BW-750-285.png}}%
\logodimen=\wd\logobox%
\hbox to \textwidth{\hss%
\begin{minipage}[t]{\logodimen}%
{\fontsize{7}{7}\selectfont\uppercase{\logofltrack{In Association With}}\par}%
\vspace{5\p@}%
\centerline{\includegraphics[width=102pt,height=64pt]{VTPublishing-Logo-BW-750-285.png}}\par%
\vspace{3\p@}%
{\fontsize{7}{7}\selectfont\uppercase{\logotrack{Blacksburg, Virginia}}\par}%
\end{minipage}%
\hss}\egroup}%

\newenvironment{copyrt}{\newpage\thispagestyle{empty}\bgroup%
\null\vspace{-2.5\baselineskip}\vspace{-50\p@}%
\raggedright%
\normalsize\normalfont%
\setlength{\parskip}{8\p@}}{\par\egroup}%

\newcommand\mainmatter{%
  \clearpage%
  \@mainmattertrue%
  \pagenumbering{arabic}%
\pagestyle{headings}%
\setlength{\parskip}{12\p@}%
}%

\newcommand\backmatter{%
  \clearpage
  \@mainmatterfalse}

\def\part{\clearpage\thispagestyle{empty}\global\@topnum\z@
  \secdef\@part\@spart}

\def\@part[#1]#2{%
  \ifnum \c@secnumdepth >\m@ne
    \refstepcounter{part}
%    \typeout{^^J\@chapapp\space\arabic{chapter}:}
    \addcontentsline{toc}{part}{\protect\numberline{\thepart}#2}
  \else
    \addcontentsline{toc}{part}{#2}
  \fi
  \@makeparthead{#2}
  \@afterindentfalse
  \@afterheading}

\def\@spart#1{%
  \thispagestyle{empty}
  \@makesparthead{#1}
  \@afterindentfalse
  \@afterheading}

\def\chapter{\clearpage\thispagestyle{plain}\global\@topnum\z@
  \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{%
  \ifnum \c@secnumdepth >\m@ne
    \refstepcounter{chapter}
    \typeout{^^J\@chapapp\space\arabic{chapter}:}
\ifAppendix%
    \addcontentsline{toc}{appendix}{\protect\numberline{\thechapter}#2}
\else
    \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#2}\fi
  \else
    \addcontentsline{toc}{chapter}{#2}
  \fi
  \chaptermark{#1}
  \@makechapterhead{#2}
  \@afterindentfalse
  \@afterheading}

\def\@schapter#1{%
  \chaptermark{#1}
  \thispagestyle{empty}
  \@makeschapterhead{#1}
  \@afterindentfalse
  \@afterheading}

\def\@makeparthead#1{%
  \begingroup
    \parindent\z@
    \vspace*{17\p@}%
    \hbox to \hsize{\hss\fontsize{16}{18}\selectfont\partname~\thepart\hss}\par%
        \removelastskip\nointerlineskip\vskip17.5pt\nointerlineskip%
    {\fontsize{15}{17}\fontseries{b}\selectfont\centering\uppercase{#1}\par}%
\removelastskip\nointerlineskip\vspace{23pt}\nointerlineskip%
  \endgroup
  \@afterindentfalse
  \@afterheading
\clearemptydoublepage%
}

\let\@author\@empty%
\def\author#1{\def\@author{#1}}%

\let\@subtitle\@empty%
\def\subtitle#1{\def\@subtitle{#1}}%

\def\@makechapterhead#1{%
  \begingroup%
    \parindent\z@\raggedright%
\null\vspace{-78\p@}%
\vspace{-\parskip}%
\rule{\textwidth}{2\p@}\par%
\vspace{58.65\p@}%
\hbox{%
\adjustbox{valign=t}{%
\begin{minipage}[t]{109\p@}%
\vspace{5\p@}%
\rule{109\p@}{0.5\p@}\par%
\normalsize\textsf{\textbf{\MakeUppercase\chaptername~\thechapter}}%
\end{minipage}}\ignorespaces\hspace{8\p@}\ignorespaces%
\adjustbox{valign=t}{%
\begin{minipage}[t]{330\p@}%
\vbox to 211\p@{%
    {\raggedright\fontsize{24}{30}\fontshape{it}\selectfont#1\vphantom{y}\par}%
\vfill}%
\end{minipage}}%
}\par%
\vspace{-\parskip}%
\rule{\textwidth}{0.5\p@}\par%
\vspace{-\parskip}%
\vspace{8\p@}%
  \endgroup
  \@afterindentfalse
  \@afterheading}

%%%%Appendix Chapter Title
\def\@makeappchapterhead#1{%
  \begingroup%
    \parindent\z@\raggedright%
\null\vspace{-78\p@}%
\vspace{-\parskip}%
\rule{\textwidth}{2\p@}\par%
\vspace{58.65\p@}%
\hbox{%
\adjustbox{valign=t}{%
\begin{minipage}[t]{109\p@}%
\vspace{5\p@}%
\rule{109\p@}{0.5\p@}\par%
\normalsize\textsf{\textbf{\MakeUppercase\chaptername~\thechapter}}%
\end{minipage}}\ignorespaces\hspace{8\p@}\ignorespaces%
\adjustbox{valign=t}{%
\begin{minipage}[t]{330\p@}%
\vbox to 211\p@{%
    {\raggedright\fontsize{24}{30}\fontshape{it}\selectfont#1\vphantom{y}\par}%
\vfill}%
\end{minipage}}%
}\par%
\vspace{-\parskip}%
\rule{\textwidth}{0.5\p@}\par%
\vspace{-\parskip}%
\vspace{8\p@}%
  \endgroup
  \@afterindentfalse
  \@afterheading}


  \def\@makeschapterhead#1{%
  \begingroup\parindent\z@\raggedright%
\null\vspace{-64\p@}%
    {\fontsize{18}{22}\fontseries{b}\selectfont#1\vphantom{y}\par}%
\vspace*{21\p@}%
  \endgroup%
  \@afterindentfalse%
  \@afterheading}%

% Section macros

% \@sect redefined to write the long entry to the TOC
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \ifnum#2=1
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M #8 \@@par\vspace{-\parskip}}%
    \endgroup
    \else
    \begingroup
      #6{%
        \@hangfrom{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M #8 \@@par\vspace{-\parskip}}%
    \endgroup\fi
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #8}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8}%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #8}}%
  \fi
  \@xsect{#5}}


\newdimen\bsecdimen%
%\bsecdimen=20pt%%%
\bsecdimen=8pt%%%

\newdimen\asecdimen%
\asecdimen=9pt%

\renewcommand{\@seccntformat}[1]{{\csname the#1\endcsname\ignorespaces\hspace{9pt}\ignorespaces}}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-\bsecdimen \@plus -4\p@}%
                                   {\asecdimen}
{\fontsize{14}{16}\fontshape{it}\selectfont\raggedright}}

\newdimen\bsubsecdimen% B and C level above spaces are same
%\bsubsecdimen=11pt%%
\bsubsecdimen=0.01pt%%

\newdimen\asubsecdimen%
\asubsecdimen=0.01pt%%

\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-\bsubsecdimen \@plus -3\p@}%
                                     {\asubsecdimen}%
%{\fontsize{11}{13}\fontseries{b}\selectfont\raggedright}}
{\fontsize{11}{13}\bfseries\raggedright}}

\newdimen\bsubsubsecdimen% B and C level above spaces are same
%\bsubsubsecdimen=12pt%%
\bsubsubsecdimen=0.01pt%%

\newdimen\asubsubsecdimen%
\asubsubsecdimen=5pt%%

\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-\bsubsecdimen \@plus -3\p@}%
                                     {-\asubsubsecdimen}%
%{\normalsize\fontseries{b}\selectfont\raggedright}}
{\normalsize\bfseries\raggedright}}

\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                     {-6\p@ \@plus -3\p@}%
                                     {-1em}%
{\fontsize{10.5}{12}\selectfont\raggedright\itshape}}

\newcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}%
                                     {-6\p@ \@plus -3\p@}%
                                     {-1em}%
{\fontsize{9}{11}\selectfont\raggedright\itshape}}

\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{6\p@}
\setlength\belowcaptionskip{\z@}

\long\def\@makecaption#1#2{\vspace{\abovecaptionskip}%
\normalsize\RaggedRight%
  \setbox\@tempboxa=\hbox{{\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}#2}%
  \ifdim\wd\@tempboxa>\hsize
  {\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}\selectfont\RaggedRight#2%
  \else%
    \hbox to \hsize{\hss{\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}\selectfont#2\hss}%
  \fi}%

\long\def\@makeshortcaption#1#2{\vspace{\abovecaptionskip}%
\normalsize\RaggedRight%\RaggedLeft%
  \setbox\@tempboxa=\hbox{{\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}#2}%
  \ifdim\wd\@tempboxa>\hsize
  {\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}\selectfont\RaggedRight#2%
  \else%
    \hbox to \hsize{\hss{\small\textsf{\textbf{#1}}}\ignorespaces\hspace{10\p@}\ignorespaces\bfseries\mathversion{bold}\selectfont#2}%
  \fi}%

\newbox\figwdbox%
\newdimen\figwddimen%

\newlength{\floataboveskip}%
\newlength{\floatbelowskip}%
\setlength{\floataboveskip}{\z@}%
\setlength{\floatbelowskip}{\z@}%

%%29pc and 20pc
\newcommand{\processfigure}[3][tbph!]{\bgroup\parindent\z@%
\setbox\figwdbox=\hbox{#2}%
\figwddimen=\wd\figwdbox%
\begin{figure}[#1]%
\vspace{\floataboveskip}%
\ifdim\figwddimen=20pc%
\let\@makecaption\@makeshortcaption%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
%\checkoddpage\ifoddpage%
%\begin{minipage}[b]{\figwddimen}%
%{#2}%
%\end{minipage}%
%\ignorespaces\hspace{12\p@}\ignorespaces%
%\begin{minipage}[b]{\textwidth}%
%{\RaggedRight#3}%
%\vspace{67\p@}%
%\end{minipage}%
%\else%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
%\vspace{67\p@}%
\end{minipage}\ignorespaces\hspace{12\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}%
%\fi%end even page
\else%
\ifdim\figwddimen<20.5pc%
\let\@makecaption\@makeshortcaption%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
%\checkoddpage\ifoddpage%
%\begin{minipage}[b]{\figwddimen}%
%{#2}%
%\end{minipage}%
%\ignorespaces\hspace{12\p@}\ignorespaces%
%\begin{minipage}[b]{\textwidth}%
%{\RaggedRight#3}%
%\vspace{67\p@}%
%\end{minipage}%
%\else%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
%\vspace{67\p@}%
\end{minipage}\ignorespaces\hspace{12\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}%
%\fi%end even page  %end of condtion1 "=" or "less than 20pc"
\else%
\ifdim\figwddimen=29pc%
\let\@makecaption\@makeshortcaption%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
%\checkoddpage\ifoddpage%
%\begin{minipage}[b]{\figwddimen}%
%{#2}%
%\end{minipage}%
%\ignorespaces\hspace{12\p@}\ignorespaces%
%\begin{minipage}[b]{\textwidth}%
%{\RaggedRight #3}%
%\end{minipage}%
%\else%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
\end{minipage}\ignorespaces\hspace{12\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}%
%\fi%end even page
\else%
\ifdim\figwddimen<29.5pc%
\let\@makecaption\@makeshortcaption%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
%\checkoddpage\ifoddpage%
%\begin{minipage}[b]{\figwddimen}%
%{#2}%
%\end{minipage}%
%\ignorespaces\hspace{12\p@}\ignorespaces%
%\begin{minipage}[b]{\textwidth}%
%{\RaggedRight#3}%
%\end{minipage}%
%\else%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
\end{minipage}\ignorespaces\hspace{12\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}%
%\fi%end even page  %end of condtion2 "=" or "less than 29pc"
\else%
\ifdim\figwddimen>29.5pc%
{\centerline{#2}}{#3}
\fi\fi\fi\fi\fi%
\vspace{\floatbelowskip}%
\end{figure}%
\egroup}%

%\newcommand{\oddpgsidecapfigure}[3][tbph!]{\bgroup\parindent\z@%
%\setbox\figwdbox=\hbox{#2}%
%\figwddimen=\wd\figwdbox%
%\begin{figure}[#1]%
%\vspace{\floataboveskip}%
%\addtolength{\textwidth}{-\figwddimen}%
%\addtolength{\textwidth}{-12\p@}%
%\begin{minipage}[b]{\figwddimen}%
%{#2}%
%\end{minipage}%
%\ignorespaces\hspace{12\p@}\ignorespaces%
%\begin{minipage}[b]{\textwidth}%
%{\RaggedRight#3}%
%\vspace{67\p@}%
%\end{minipage}%
%\vspace{\floatbelowskip}%
%\end{figure}%
%\egroup}%

\newcommand{\evenpgsidecapfigure}[3][tbph!]{\bgroup\parindent\z@%
\setbox\figwdbox=\hbox{#2}%
\figwddimen=\wd\figwdbox%
\begin{figure}[#1]%
\vspace{\floataboveskip}%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
%\vspace{67\p@}%
\end{minipage}\ignorespaces\hspace{12\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}%
\vspace{\floatbelowskip}%
\end{figure}%
\egroup}%

\let\oddpgsidecapfigure\evenpgsidecapfigure%

\newcommand{\splprocessfigure}[3][tbph!]{\bgroup\parindent\z@%
\setbox\figwdbox=\hbox{#2}%
\figwddimen=\wd\figwdbox%
\begin{figure}[#1]%
\vspace{\floataboveskip}%
\addtolength{\textwidth}{-\figwddimen}%
\addtolength{\textwidth}{-12\p@}%
\hbox to 37.5pc{\hfill%
\begin{minipage}[c]{\textwidth}%
{\RaggedRight#3}%
\end{minipage}\ignorespaces\hspace{-120\p@}\ignorespaces%
\begin{minipage}[c]{\figwddimen}%
{#2}%
\end{minipage}\hfill}%
\vspace{\floatbelowskip}%
\end{figure}%
\egroup}

\RequirePackage{wrapfig}%

%%%


\long\def\@tablecaption#1#2{\normalfont\normalsize\bfseries\mathversion{bold}\selectfont%
{\small\textbf{\textsf{#1}}}\hspace{1em}{#2\strut}\par
  \vspace{\belowcaptionskip}}

\newcommand{\notename}{Notes{\em:}}

% Top rule
\def\toprule{\noalign{\vskip3pt}
\noalign{\ifnum0=`}\fi\hrule
  \@height 0.5\p@ % <- rule height
  \hrule \@height 4pt \@width 0pt
  \futurelet\@tempa\@xhline
  \noalign{\vskip-.7pt}
  }
% Bottom rule
\def\botrule{\noalign{\ifnum0=`}\fi \hrule \@height 3.2pt \@width 0pt
  \hrule \@height 0.5\p@ % <- rule height
  \futurelet\@tempa\@xhline
  \noalign{\vskip1pt}}
% Middle rule
\def\midrule{\noalign{\ifnum0=`}\fi \hrule \@height 3.25pt \@width 0pt
  \hrule \@height 0.5pt % <- rule height
  \hrule \@height 5pt \@width 0pt \futurelet \@tempa\@xhline}
% Bridge rule
\def\hrulefill{\leavevmode\leaders\hrule height 0.5pt\hfill\kern\z@}


\def\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\def\figure{\@float{figure}}
\def\endfigure{\end@float}
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}
\def\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\def\table{\let\@makecaption\@tablecaption\@float{table}}
\let\endtable\end@float
\@namedef{table*}{\let\@makecaption\@tablecaption\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}

\newdimen\tempdime
\newbox\temptbox
\newenvironment{processtable}[3]{\setbox\temptbox=\hbox{\fontsize{10}{12}\selectfont{#2}}%
\tempdime\wd\temptbox\@processtable{#1}{#2}{#3}{\tempdime}}
{\relax}

\newenvironment{@processtable}[4]{%
\setbox4=\hbox to \textwidth{\vbox{\begin{center}\begin{minipage}[t]{#4}%
\fontsize{10}{12}\selectfont%
\caption{#1}{#2}%
\vskip4pt\noindent
\parbox{#4}{\fontsize{8}{10}\selectfont #3\par}%
\end{minipage}\end{center}}}%
\box4}
{\relax}

\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}%

\newcommand{\colhead}[1]{\bgroup\mathversion{bold}\bfseries#1\egroup}%

% ******************************
% List numbering and lettering *
% ******************************
\def\labelenumi{{\rm\arabic{enumi}.}}
\def\theenumi{\arabic{enumi}}
\def\labelenumii{{\rm \roman{enumii}.}}
\def\theenumii{a\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{({\rm\roman{enumiii}})}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{({\rm\Alph{enumiv}})}
\def\theenumiv{\Alph{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}

\newcommand{\alphlist}{%
\def\labelenumi{{\rm\alph{enumi}}}%
\def\theenumi{\alph{enumi}}%
}

\newcommand{\Alphlist}{%
\def\labelenumi{{\rm\Alph{enumi}}}%
\def\theenumi{\Alph{enumi}}%
}

\newcommand{\romanlist}{%
\def\labelenumi{{\rm\roman{enumi}}}%
\def\theenumi{\roman{enumi}}%
}

\newcommand{\Romanlist}{%
\def\labelenumi{{\rm\Roman{enumi}}}%
\def\theenumi{\Roman{enumi}}%
}

\newcommand{\smallbullet}{} % for safety
\DeclareRobustCommand\smallbullet{%
  \mathord{\mathpalette\smallbullet@{0.8}}%
}
\newcommand{\smallbullet@}[2]{%
  \vcenter{\hbox{\scalebox{#2}{$\m@th#1\bullet$}}}%
}
\def\labelitemi{$\smallbullet$}
\def\labelitemii{$\bullet$}
\def\labelitemiii{$\bullet$}
\def\labelitemiv{$\bullet$}


\setlength\leftmargini   {0pt}%
\leftmargin  \leftmargini
\setlength\leftmarginii  {\z@}
\setlength\leftmarginiii {\z@}
\setlength\leftmarginiv  {\z@}
\setlength\leftmarginv   {\z@}
\setlength\leftmarginvi  {\z@}

\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}

\def\@listi{\leftmargin\leftmargini
            %\parsep 4\p@ \@plus2\p@ \@minus\p@
            \topsep 6\p@ \@plus2\p@ \@minus4\p@
            %\itemsep4\p@ \@plus2\p@ \@minus\p@
            }
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep   7\p@% 4\p@ \@plus2\p@ \@minus\p@
              \parsep    \z@ %\@plus\p@  \@minus\p@
              %\itemsep   \parsep
}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

\def\enumargs{%
   \partopsep     \z@
%   \itemsep       \z@
    \topsep      0\p@
   \itemsep       5\p@
   \parsep        \z@
   \labelsep      0.5em
   \rightmargin   \z@
   \listparindent \parindent
   \itemindent    \z@}

\def\enumerate{%
    \@ifnextchar[{\@numerate}{\@numerate[0.]}}

\def\@numerate[#1]{%
     \ifnum \@enumdepth >3 \@toodeep\else
     \advance\@enumdepth \@ne
     \edef\@enumctr{enum\romannumeral\the\@enumdepth}
     \list{\csname label\@enumctr\endcsname}{%
       \enumargs
       \setlength{\leftmargin}{\csname leftmargin\romannumeral\the\@enumdepth\endcsname}
       \usecounter{\@enumctr}
       \settowidth\labelwidth{#1}
       \addtolength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
%       \def\makelabel##1{\hss\llap{\sansbold{##1}}}}%
       \def\makelabel##1{\hss\llap{{##1}}}}%
     \fi
   }
\let\endenumerate\endlist


% Changes to the list parameters for itemize
\def\itemargs{%
   \partopsep     \z@
   \itemsep       5\p@
   \parsep        \z@
    \topsep      0\p@
%   \labelsep      0.5em
   \labelsep      10\p@%
   \rightmargin   \z@
   \listparindent \parindent
   \itemindent    \z@}

\def\itemize{%
   \@ifnextchar[{\@itemize}{\@itemize[$\bullet$]}}

\def\@itemize[#1]{%
     \ifnum \@itemdepth >3 \@toodeep\else
     \advance\@itemdepth \@ne
     \edef\@itemctr{item\romannumeral\the\@itemdepth}
     \list{\csname label\@itemctr\endcsname}{%
       \itemargs
       \setlength{\leftmargin}{\csname leftmargin\romannumeral\the\@itemdepth\endcsname}
       \settowidth\labelwidth{#1}
       \addtolength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
       \def\makelabel##1{\hss \llap{\sansbold{##1}}}}%
     \fi
   }
\let\enditemize\endlist

\def\unargs{%
   \partopsep     \z@%
   \itemsep       5\p@%
   \labelwidth    \z@%
   \topsep        0\p@
   \parsep        \z@%
   \labelsep      \z@%
   \rightmargin   \z@%
   \listparindent \parindent%
   \leftmargin 2\parindent%
   \itemindent -1\parindent}%

\newenvironment{unlist}{%
\begin{list}{}{%
  \unargs%
\let\makelabel\unlistlabel%
  }}{\end{list}}
\newcommand*\unlistlabel[1]{%\hspace\labelsep
                                \normalfont\bfseries #1}


\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                %\normalfont\bfseries #1}
                                \normalfont #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}%
               {\list{}{\leftmargin15pt\listparindent\parindent\topsep-3\p@}%
                \normalsize\itemindent-5\p@\listparindent\parindent%
                \item\relax}%
               {\endlist}%

\skip\@mpfootins = \skip\footins
\fboxsep=6\p@
\fboxrule=1\p@

%%Boxed text%%
\setlength{\FrameSep}{10\p@}%
\setlength{\FrameRule}{0.5\p@}%

\let\boxedtext\framed%
\let\endboxedtext\endframed%

\definecolor{shadecolor}{cmyk}{0,0,0,0.20}%
\newenvironment{shdadeboxedtext}{\begingroup%
\begin{shaded}%
\advance\hsize by -2\FrameSep%
\@afterheading%
}{\end{shaded}\endgroup}%

%%
\newenvironment{thebibliography}[1]{%
 \section{\refname}%
\nopagebreak\vspace*{5\p@}\nopagebreak%
%\addcontentsline{toc}{chapter}{\refname}%
\normalsize\list{}{%
  \usecounter{enumi}%
  \leftmargin20\p@\itemindent-20\p@\parsep\z@%
    \itemsep5\p@%
 }}%
{\endlist}%
\RequirePackage[numbers]{natbib}%
\def\@biblabel#1{\relax}%
\renewenvironment{thebibliography}[1]{%
 \section{\refname}%
\nopagebreak\vspace*{5\p@}\nopagebreak%
%\addcontentsline{toc}{chapter}{\refname}%
\normalsize\list{}{%
  \usecounter{enumi}%
  \leftmargin20\p@\itemindent-25\p@\parsep\z@%
    \itemsep5\p@%
 }}%
{\endlist}%


\RequirePackage{multicol}

\newif\if@restonecol
\def\theindex{\@mainmatterfalse
  \addcontentsline{toc}{chapter}{Index}
  \columnseprule \z@
  \columnsep 3pc\begin{multicols}{2}[\chapter*{\indexname}]
                                %Subject or Author here
  \markboth{\indexname}{\indexname}%
  \parskip\z@ plus .3\p@\relax\let\item\idxitem\fontsize{8}{9.5}\selectfont}
\def\idxitem{\par\noindent\raggedright\hangindent 3em}
\def\subitem{\par\noindent\raggedright\hangindent 4em\hspace*{1em}}
\def\subsubitem{\par\noindent\raggedright\hangindent 5em\hspace*{2em}}
\def\endtheindex{\end{multicols}}
\def\indexspace{\par \vskip 12\p@ plus 5\p@ minus 3\p@\relax}

% ***********
% Footnotes *
% ***********

%\usepackage[hang,splitrule,perpage]{footmisc}%
\usepackage[splitrule]{footmisc}%
  \def\splitfootnoterule{\kern-3\p@ \hbox to 0.5\textwidth{\hrulefill} \kern3.6\p@}

\renewcommand\footnoterule{\hbox to 6pc{\hrulefill}\par\removelastskip\nointerlineskip\vskip3.5pt\nointerlineskip}%
%\footnotemargin12\p@%\relax
\renewcommand\footnotelayout{\fontsize{8.8}{10.2}\selectfont}
\usepackage{scrextend}
%\deffootnote{5mm}{5mm}{\makebox[5mm][l]{%\phantom{99}\llap
%{\thefootnotemark.}}}


\renewcommand\@makefntext[1]{%
    \fontsize{8.8}{9}\selectfont\def\@textsuperscript{}%\global\baselineskip=9pt%
    \parindent10pt%
    \noindent\ignorespaces%\hspace*{8.8pt}\ignorespaces%
    \hbox{\@makefnmark.}\ignorespaces\hspace*{8.8pt}\ignorespaces#1}
\def\@makefnmark{\hbox{\smash{\@textsuperscript{\normalfont\@thefnmark}}}}

\@addtoreset{footnote}{chapter}%

%Math parameters

\setlength{\jot}{5\p@}

\def\frenchspacing{\sfcode`\.\@m \sfcode`\?\@m \sfcode`\!\@m
  \sfcode`\:\@m \sfcode`\;\@m \sfcode`\,\@m}

\def\@mathmargin{1.5pc}
\vbadness=9999
\tolerance=9999
\doublehyphendemerits=10000
\doublehyphendemerits 640000   % corresponds to badness 800
\finalhyphendemerits  1000000  % corresponds to badness 1000

% Table and array parameters
\setlength\arraycolsep{.5em}
\setlength\tabcolsep{.5em}
\setlength\arrayrulewidth{.25pt}
\setlength\doublerulesep{2.5pt}
\renewcommand\arraystretch{1}

\def\@dottedtocline#1#2#3#4#5{%
  \renewcommand\@dotsep{1000}
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {#4}\nobreak
     \leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
     \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalcolor\normalsize #5}%
     \par}%
  \fi}


\def\@authordottedtocline#1#2#3#4#5{%
    \renewcommand\@dotsep{1000}
  \ifnum #1>\c@tocdepth \else
    \vskip 0\p@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {\it #4}\nobreak
     \leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
  \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalcolor\normalsize}%
     \par}%
\vspace{6pt}
  \fi}


\def\@chapterdottedtocline#1#2#3#4#5{%
  \begingroup
\def\numberline##1{{\MakeUppercase\chaptername~##1}\ignorespaces\hspace{10.85\p@}\ignorespaces}%
  \renewcommand\@dotsep{5}%
  \ifnum #1>\c@tocdepth \else
    \vskip 25pt \@plus 1pt\goodbreak
    {\leftskip #2\relax   \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
\fontsize{14}{16}\selectfont%
     {\def\break{ }#4}\nobreak
     \leaders\hbox{$\m@th         \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalcolor{\bf#5}}%
     \par\vspace{8\p@}}%
  \endgroup
  \fi}

\def\@secdottedtocline#1#2#3#4#5{%
  \renewcommand\@dotsep{1000}
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ %\@plus.2\p@
    {\leftskip #2\relax %\rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {#4}\ignorespaces\quad\ignorespaces{\normalcolor\normalsize\bf #5}%
%     \leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
%        mu$}\hfill
%     \nobreak
%     \hb@xt@\@pnumwidth{\hfil\normalcolor\normalsize\bf #5}%
     \par}%
  \fi}

\def\@subsecdottedtocline#1#2#3#4#5{%
  \renewcommand\@dotsep{1000}
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ %\@plus.2\p@
    {\leftskip #2\relax \rightskip\z@ plus1fill %\rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
     {\it#4}\nobreak\ignorespaces\quad\ignorespaces\nobreak{\normalcolor\normalsize\fontseries{b}\fontshape{it}\selectfont#5}%
%     \leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
%        mu$}\hfill
%     \nobreak
%     \hb@xt@\@pnumwidth{\hfil\normalcolor\normalsize\bf #5}%
     \par}%
  \fi}

\def\@appdottedtocline#1#2#3#4#5{%
  \begingroup
\def\numberline##1{\hb@xt@90\p@{\uppercase{Appendix~##1}\hfil}}
  \renewcommand\@dotsep{5}%
  \ifnum #1>\c@tocdepth \else
    \vskip 25pt \@plus 1pt\goodbreak
    {\leftskip #2\relax   \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
\fontsize{14}{16}\selectfont%
     {\def\break{ }#4}\nobreak
     \leaders\hbox{$\m@th         \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalcolor{\bf#5}}%
     \par\vspace{8\p@}}%
  \endgroup
  \fi}

\def\@partdottedtocline#1#2#3#4#5{%
  \begingroup
  \def\numberline##1{\hbox{{Part ##1\quad}}}
  \renewcommand\@dotsep{1000}
  \ifnum #1>\c@tocdepth \else
    \vskip 16pt \@plus 1pt\goodbreak
    {\leftskip #2\relax   \rightskip \@tocrmarg \parfillskip -\rightskip
     \parindent #2\relax\@afterindenttrue
     \interlinepenalty\@M
     \leavevmode
     \@tempdima #3\relax
     \null\nobreak\hskip -\leftskip
     {\fontsize{11}{13}\selectfont\bfseries #4}\nobreak
     \leaders\hbox{$\m@th         \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
     \nobreak
     \hb@xt@\@pnumwidth{\hfil\normalcolor\normalsize {\fontsize{11}{13}\selectfont\bfseries #5}}%
     \par}%
  \endgroup
  \fi}

\newcommand\@pnumwidth{20\p@}%
\newcommand\@tocrmarg{2.55em plus 1fil}
\newcommand\@dotsep{1}             %%% <-- very large value so we don't get any dots

  \def\@makestocchapterhead#1{%
  \begingroup\parindent\z@\raggedright%
\null\vspace{-64\p@}%
    {\fontsize{18}{22}\fontseries{b}\selectfont#1\vphantom{y}\par}%
\vspace*{1.5\p@}%
  \endgroup%
  \@afterindentfalse%
  \@afterheading}%

\newcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
%\let\@makeschapterhead\@makestocchapterhead%
\@makestocchapterhead{Table of Contents}%
\thispagestyle{empty}%
\vspace{-20\p@}%
\markboth{Contents}{Contents}%
    \chaptermark{Contents}       %% <-- check the running heads in all \chapter*
    \@starttoc{toc}%                    %%      Code manually with \markboth{}{}
    \if@restonecol\twocolumn\fi
    }

\setcounter{tocdepth}{3}%

\newcommand*\l@book[1]{{\vspace{\bigskipamount}\vspace{\medskipamount}\noindent{\large\bfseries #1}}}
\newcommand*\l@part{\@partdottedtocline{0}{0em}{60pt}}
\newcommand*\l@prelims{\@dottedtocline{0}{0em}{20pt}}
\newcommand*\l@chapter{\@chapterdottedtocline{1}{0pt}{21pt}}
\newcommand*\l@section{\@secdottedtocline{2}{50pt}{27pt}}
\newcommand*\l@author{\@authordottedtocline{2}{15pt}{27pt}}
\newcommand*\l@subsection{\@subsecdottedtocline{3}{77pt}{29pt}}
\newcommand*\l@subsubsection{\@dottedtocline{4}{76.5pt}{35pt}}
\newcommand*\l@paragraph{\@dottedtocline{5}{114pt}{40pt}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{154.5pt}{50pt}}
\newcommand*\l@appendix{\@appdottedtocline{1}{0pt}{21pt}}
\newcommand*\l@appsection{\@dottedtocline{2}{17pt}{23pt}}
\newcommand*\l@endmatter{\@chapterdottedtocline{2}{0em}{0em}}

\flushbottom%
\frenchspacing%
\ps@headings%
\onecolumn%
\fnbelowfloat%

%\graphicspath{{../97968_art/},{../97968_art/97968_ornaments/}}%

\hyphenation{Figure Figures Table Tables Section Sections}

% Reset eqnarray to avoid to avoid \arraycolsep between
% columns. Take from a post on comp.text.tex
% by Fred Bartlett <fbartlet@optonline.net>
\newif\if@alignpoint
\def\eqnarray{%
  \stepcounter{equation}%
  \def\@currentlabel{\p@equation\theequation}%
  \global\@eqnswtrue
  \m@th
  \global\@eqcnt\z@
  \tabskip\@centering
  \let\\\@eqncr
  $$\everycr{}\halign to\displaywidth\bgroup
    \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
  &\global\@eqcnt\@ne\hfil\setbox\z@\hbox{$\displaystyle{{}##{}}$}%
    \global\ifdim\wd\z@>\z@\@alignpointtrue\else\@alignpointfalse\fi
    \box\z@\hfil
  &\global\@eqcnt\tw@$\displaystyle{\if@alignpoint\else{}\fi##}$\hfil
    \tabskip\@centering
  &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
         \tabskip\z@skip
  \cr}



%%%Appendix%%
\newcounter{appcounter}%
\setcounter{appcounter}{0}
%\refstepcounter{appcounter}%
\renewcommand{\theappcounter}{\Alph{appcounter}}%

\def\appendix{\par%
\clearpage
\Appendixtrue%
\setcounter{page}{1}%
\let\@makechapterhead\@makeappchapterhead%
  \renewcommand{\thechapter}{\Alph{chapter}}%
  \renewcommand{\thepage}{\thechapter-\arabic{page}}%
        \setcounter{chapter}{0}
        \setcounter{section}{0}
        \setcounter{table}{0}
        \setcounter{subsection}{0}
\def\chaptername{Appendix}}

\newtheoremstyle{common}
    {6pt plus 5\p@ minus 2\p@}% above space (default)
    {6pt plus 5\p@ minus 2\p@}% below space
    {\itshape}% body
    {0em}% indent
    {\bfseries}% head
    {}% punct
    {.5em}% space
%    {}% custom
{\thmname{#1}\thmnumber{\@ifnotempty{#1}{ }{#2}}%
 \thmnote{ {(#3)}}}% theorem head
\theoremstyle{common}

%%%With Chapter Number
\newtheorem{theorem}{Theorem}%
\newtheorem{corollary}{Corollary}%
\newtheorem{lemma}{Lemma}%
\newtheorem{definition}{Definition}%
\newtheorem{proposition}{Proposition}%
\newtheorem{assumption}{Assumption}%
\newtheorem{remark}{Remark}%

%%%Proof%%
%%For Proof definition inside the amsthm.sty%%%


%%%Proof*%%
\newenvironment{proof*}[1][\proofname]{\par
  \normalfont
  \topsep6\p@\@plus6\p@ \trivlist
  \item[\hskip\labelsep\bfseries\itshape
    #1\@addpunct{.}]\ignorespaces
}{%
  %\qed
  \endtrivlist
}

%\usepackage{arydshln}

%%Example%%
\newcounter{example}%
\setcounter{example}{0}%
\renewcommand{\theexample}{\thechapter.\arabic{example}}%

\newenvironment{example}[1][\relax]{\bgroup%
\def\eqsansbold##1{{\textbf{\textsf{##1}}}}%
\def\qed{\rule{5pt}{5pt}}%
\refstepcounter{example}%
\vspace{1\baselineskip plus3\p@}%
\noindent\textbf{Example~\theexample\ignorespaces\enskip\ignorespaces#1\vphantom{y}}\\[-2\p@]%
\nopagebreak\rule{\textwidth}{2\p@}\\[8\p@]%
\nopagebreak%
}%{\ignorespaces\quad\qed\par\vspace{1\baselineskip plus3\p@}%
{\ignorespaces\hfill\qed\par\vspace{1\baselineskip plus3\p@}%
\egroup}%


\newenvironment{example*}[1][\relax]{\bgroup%
\def\eqsansbold##1{{\textbf{\textsf{##1}}}}%
\def\qed{\rule{5pt}{5pt}}%
\refstepcounter{example}%
\vspace{1\baselineskip plus3\p@}%
\noindent\textbf{Example~\theexample\ignorespaces\enskip\ignorespaces#1\vphantom{y}}\\[-2\p@]%
\nopagebreak\rule{\textwidth}{2\p@}\\[8\p@]%
\nopagebreak%
}{\par\vspace{1\baselineskip plus3\p@}%
\egroup}%

\@addtoreset{example}{chapter}%

%%exercise
\def\exerenumargs{%
   \partopsep     \z@
%   \itemsep       \z@
   \itemsep       5\p@
   \parsep        \z@
   \labelsep      10\p@
   \rightmargin   \z@
   \listparindent \parindent
   \itemindent    5\p@}

\def\exerenumerate{%
    \@ifnextchar[{\@exernumerate}{\@exernumerate[0.]}}

\def\@exernumerate[#1]{%
     \ifnum \@enumdepth >3 \@toodeep\else
     \advance\@enumdepth \@ne
     \edef\@enumctr{enum\romannumeral\the\@enumdepth}
     \list{\csname label\@enumctr\endcsname}{%
       \exerenumargs
       \setlength{\leftmargin}{\csname leftmargin\romannumeral\the\@enumdepth\endcsname}
       \usecounter{\@enumctr}
%       \settowidth\labelwidth{#1}
%       \addtolength{\leftmargin}{\labelwidth}
%       \addtolength{\leftmargin}{\labelsep}
       \def\makelabel##1{{\sansbold{##1}}}}%
     \fi
   }
\let\endexerenumerate\endlist

\def\@exerlistii{\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep   7\p@% 4\p@ \@plus2\p@ \@minus\p@
              \parsep    \z@ %\@plus\p@  \@minus\p@
              %\itemsep   \parsep
\let\sansbold\relax%
}

\newenvironment{exercise}{\bgroup%
\let\enumerate\exerenumerate%
\let\endenumerate\endexerenumerate%
\let\@listii\@exerlistii%
}{\egroup}%


%%

\let\cal\mathcal

\makeindex

% Author queries
\newcommand{\query}[2][0pt]{%
  \marginpar{\vspace*{#1}%
    \fbox{\parbox{6pc}{%
     \raggedright\small
        AQ: #2}}}}

\def\bond{\hbox{${-}$}}%
\def\dbond{\smash{\raisebox{0.75\p@}{\hbox{${=}$\!\!\!\!${=}$}}}}%
\def\tbond{\smash{\raisebox{0.75\p@}{\hbox{${\equiv}$\!\!\!\!${\equiv}$}}}}%

\AtBeginDocument{%
\thinmuskip=3mu%
\medmuskip=3mu%
\thickmuskip=3mu%
}%


%%Don't change this, because of to align the PDI and the text%%
\usepackage{atbegshi}%%
\AtBeginShipout{%
\enlargethispage{-2.55pt}%This value is fixed one, don't modify
 }%
%%%

\RequirePackage[bookmarks=true,colorlinks=false,breaklinks,linkcolor=black,citecolor=black,urlcolor=black,hidelinks,%
pdftitle={Aerospace Structures},pdfauthor={Eric Raymond Johnson},pdfkeywords={Virginia Tech Publishing}]{hyperref}%
%\RequirePackage{breakurl}%
\urlstyle{rm}%
\def\url@acsstyle{%
  \def\UrlSpecials{%
    \do\/{\penalty\UrlBreakPenalty\mathchar`/}%
    \do\~{\penalty\UrlBreakPenalty\mathchar`~}%
    \do\.{\penalty\UrlBreakPenalty\mathchar`.}%
    \do\,{\penalty\UrlBreakPenalty\mathchar`,}%
    \do\-{\penalty\UrlBreakPenalty\mathchar`-}%
    \do\_{\penalty\UrlBreakPenalty\mathchar`_}%
    \do\?{\penalty\UrlBreakPenalty\mathchar`?}%
    \do\#{\penalty\UrlBreakPenalty\mathchar"23}%
    \do\%{\Url@percent}%
    \do\={\penalty\UrlBreakPenalty\mathchar`=\penalty\UrlBreakPenalty}%
    \do\&{\penalty\UrlBreakPenalty\mathchar`&\penalty\UrlBreakPenalty}%
    \do\ {\Url@space}\do\^^M{\Url@space}%
    \Url@force@Tilde}%
}
\urlstyle{acs}


\AtBeginDocument{%
\def\tagform@#1{\maketag@@@{\footnotesize\sansbold{(\ignorespaces#1\unskip\@@italiccorr)}}}%
}%

\newcommand{\yenbars}{%
  \vbox{
     \hrule height.1ex width.4em
     \kern.15ex
     \hrule height.1ex width.4em
     \kern.3ex
  }%
}

\DeclareTextCommandDefault{\textyen}{%
  \vphantom{Y}%
  {\ooalign{Y\cr\hidewidth\yenbars\hidewidth\cr}}%
}

\AtBeginDocument{%
\DeclareSymbolFont{numbers}{\encodingdefault}{\rmdefault}{m}{n}
\DeclareMathSymbol{0}{\mathalpha}{numbers}{"30}
\DeclareMathSymbol{1}{\mathalpha}{numbers}{"31}
\DeclareMathSymbol{2}{\mathalpha}{numbers}{"32}
\DeclareMathSymbol{3}{\mathalpha}{numbers}{"33}
\DeclareMathSymbol{4}{\mathalpha}{numbers}{"34}
\DeclareMathSymbol{5}{\mathalpha}{numbers}{"35}
\DeclareMathSymbol{6}{\mathalpha}{numbers}{"36}
\DeclareMathSymbol{7}{\mathalpha}{numbers}{"37}
\DeclareMathSymbol{8}{\mathalpha}{numbers}{"38}
\DeclareMathSymbol{9}{\mathalpha}{numbers}{"39}
}


\makeatletter
\def\harp#1{\stackrel{\rightharpoonup}{#1}}
\newcommand{\arc}{\mbox{\large$\frown$}}
\newcommand{\bigfrown}[1]{\stackrel{\arc}{#1}}
\makeatother

\usepackage{dcolumn}
\newcolumntype{d}[1]{D{.}{.}{#1}}

\endinput
