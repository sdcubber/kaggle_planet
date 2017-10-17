% ----------------------------------------
% Class file for Homeworks - MASTAT UGent
%
% This class file can be used by the students of
% the master in statistical data analysis for 
% preparation of their homeworks. 
%
% Either save this file in the same folder as  
% your homework file, or save it in a subfolder
% you make in MiKTeX 2.8\tex\latex. This is the
% folder where all packages reside. Be sure to 
% restart texnicCenter or whatever you use before
% trying to use the file.
%
% ----------------------------------------
% author: Joris Meys
% date: 19 october 2010
% date last modified : 23 september 2015
% version: 0.2
% ----------------------------------------

% --- Class structure identification
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{homework}[2010/10/19 v0.1 Homework]

% --- Class structure declaration of options
% This class extends article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

% --- Class structure execution of options
%
\ProcessOptions \relax

\LoadClass[12pt,a4paper]{article}

% --- Class structure: Loading of extra packages
% Layout packages
\RequirePackage{fancyhdr}
\RequirePackage[a4paper,portrait]{geometry}
\RequirePackage{graphicx}
\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage{sectsty}
\RequirePackage{afterpage}

% Code packages
\RequirePackage{listings} % including source code in the files
\RequirePackage{fancyvrb}
\RequirePackage{relsize}
\RequirePackage{color}


% --- Class structure: Definitions

\long\def\@title{}
\def\title#1{\def\@title{#1}}

\long\def\@course{}
\def\course#1{\def\@course{#1}}

\def\@author{}
\def\author#1{\def\@author{#1}}

\def\@date{}
\def\date#1{\def\@date{#1}}
\global\let\@date\today

\def\@hnum{}
\def\hnum#1{\def\@hnum{#1}}
\global\let\@hnum\@empty

% --- Class structure: Code listings

\definecolor{comment}{rgb}{0.40, 0.40, 0.38}
\definecolor{background}{rgb}{0.97, 0.97, 1.00}
\definecolor{string}{rgb}{0.43, 0.27, 0.07}
\definecolor{emph}{rgb}{0.863, 0.066, 0.266}
\definecolor{number}{rgb}{0.0, 0.6, 0.6}
\definecolor{keyword}{rgb}{0.00, 0.42, 0.60}

\lstloadlanguages{R} 

%\lstset{escapeinside={(�}{�)}}   % for (*\ref{ }*) inside lstlistings (S code)

\lstset{
	language=R,
	alsodigit={.},
	deletekeywords={},
	otherkeywords={},
	classoffset=0,
	keywordstyle=\color{keyword},
	morekeywords={test1}, % Here you can add some extra functions that are not recognized
	classoffset=1,      
	morekeywords={},   % This can be used to identify user-created functions
	keywordstyle=\color{variable}\bfseries\itshape,
	classoffset=0
}

\lstset{
  basicstyle=\ttfamily\small,
  identifierstyle=\color{black},
  commentstyle=\color{comment} \itshape,
  stringstyle=\color{string}, 
  showstringspaces=false,
  columns = fullflexible,
  backgroundcolor=\color{background},
  mathescape = false,
  literate={<-}{{$\leftarrow$ }}1 {~}{{$\sim$}}1,
  fancyvrb,
  emph={T,F,TRUE,FALSE,NA},emphstyle= \color{emph},
  xleftmargin=4ex,
  xrightmargin=4ex,
  captionpos=b
}
\renewcommand{\lstlistingname}{Code}

% --- Class structure: Title header
%
\renewcommand{\maketitle}{
%	\hangindent=0.7cm
	\global\@topnum\z@ % Prevents floats to go to top of page?
	\begin{onehalfspace}
	\textbf{\Large \@title }  
	\end{onehalfspace}
	\vspace{18pt}
	\hrule
	\vskip 3pt
}

% --- Class structure: layout issues
%
\geometry{lmargin=82.0bp,rmargin=82.0bp,tmargin=82.0bp,bmargin=82.0bp}
\parindent 0pt
\parskip 10pt
\sectionfont{\large}

\fancypagestyle{firstpage}{%
	\fancyhf{} % clear all six fields
	\renewcommand{\headrulewidth}{0.7pt}
	\renewcommand{\footrulewidth}{0pt}
	\renewcommand{\headheight}{30pt}
	\fancyhead[L]{
		Report \@hnum \\ \@course 
	}
	\fancyhead[R]{
		\@date \\ \@author
	}

}
\fancypagestyle{followingpage}{%
	\fancyhf{} % clear all six fields
	\addtolength{\headheight}{3pt}
	\fancyhead[L]{
		Report \@hnum 
	}
	\fancyhead[R]{
	  \@author
	}
	\fancyfoot[C]{pg. \thepage}	
	\renewcommand{\headrulewidth}{0.7pt}	
	\renewcommand{\footrulewidth}{0.7pt}
}
\pagestyle{followingpage}
\AtBeginDocument{
	\thispagestyle{firstpage}
	\maketitle
}