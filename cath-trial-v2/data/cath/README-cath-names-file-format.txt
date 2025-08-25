CATH Names File (CNF) Format 2.0
--------------------------------
Description of each node in the CATH hierarchy for class, architecture, topology
and homologous superfamily levels.

Column 1:   Node number
Column 2:   Representative protein domain (CATH seven character domain name)
Free Text:  Node description (starting with ':' to indicate description start)
            (Note: this is not a single column)

Comment lines start with a '#' character.

Example:
--------
1                 1oaiA00       :Mainly Alpha
2                 1h8pA02       :Mainly Beta
1.10              1oaiA00       :Orthogonal Bundle
1.10.10           1i27A00       :Arc Repressor Mutant, subunit A
1.10.10.10        1i27A00       :"winged helix" repressor DNA binding domain
1.10.10.160       1pjr002       :



New to CNF Format 2.0
=====================
Zero padding has been removed from class and architecture node numbers.

Domain names are now 7 characters long rather than 6.

All nodes down to the Homologous Superfamily level are represented. Nodes
(homologous superfamily level only) that do not have a name are left blank.
