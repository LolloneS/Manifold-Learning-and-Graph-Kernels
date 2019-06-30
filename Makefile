pdf:
	pandoc -t latex --listings --include-in-header ./misc/header.tex -o Report.pdf Report.md