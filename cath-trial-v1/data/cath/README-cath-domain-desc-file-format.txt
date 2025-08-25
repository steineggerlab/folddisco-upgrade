CATH Domain Description File (CDDF) Format 2.0
----------------------------------------------
Each entry corresponds with a CATH domain for a given release of the CATH database. 
Note: Different releases of CATH may have different domains definitions.
See below for CATH domain and segment naming conventions.

Comment lines start with a '#' character.

MAXIMUM of 80 characters per line (composed of tags that always a maximum of 
10 characters and the rest of line should be no longer than 70 characters).

Tags	    Description
-----------------------
FORMAT      Format definition (CDDF2.0) and first line of each entry
DOMAIN      CATH domain identifier - seven character code (e.g. 1abcA01)
PDBID	    PDB identifier - four character code
            (currently only used in PdbSumData files)
VERSION     CATH version number
VERDATE     CATH version release date
NAME	    PDB entry description
SOURCE      PDB entry organism/source
CATHCODE    CATH superfamily code C.A.T.H e.g. 1.10.10.10
CLASS	    Text description of class level (default: 'void')
ARCH	    Text description of architecture level (default: 'void')
TOPOL	    Text description of topology level (default: 'void')
HOMOL	    Text description of homologous superfam level (default: TOPOL entry)
DLENGTH     Length of the domain sequence
DSEQH	    Domain sequence header in FASTA format (e.g. '>pdb|1abcA01')
DSEQS	    Domain sequence string in FASTA format
NSEGMENTS   Number of segments that comprise the domain (integer)
SEGMENT     Segment identifier (e.g. 1abcA01:1:2)
SRANGE      Start and stop PDB residue identifiers that define range of segment
            (e.g. START=159  STOP=202)
SLENGTH     Length of the segment sequence
SSEQH	    Segment sequence header in FASTA format (e.g. '>pdb|1abcA01:1:2')
SSEQS	    Segment sequence string in FASTA format
ENDSEG      Signifies end of segment entry
COMMENTS    Text (optional line)
//	    Signifies end of entry


Example: Cath Domain Description File (CDDF)
--------------------------------------------
FORMAT    CDDF1.0
DOMAIN    9lprA01
VERSION   3.0.0
VERDATE   04-May-2006
NAME      Alpha-lytic protease (e.c.3.4.21.12) complex with methoxysuccinyl-*ala
NAME      -*ala-*pro-*leucine boronic acid
SOURCE    (Lysobacter enzymogenes 495) cloned and expressed in (escherichia coli
SOURCE    )
CATHCODE  2.40.10.10
CLASS     Mainly Beta
ARCH      Beta Barrel
TOPOL     Thrombin, subunit H
HOMOL     Trypsin-like serine proteases
DLENGTH   87
DSEQH     >pdb|9lprA01
DSEQS     IVGGIEYSINNASLCSVGFSVTRGATKGFVTAGHCGTVNATARIGGAVVGTFAARVFPGNDRAWVSLTSA
DSEQS     QTLLLQPILSQYGLSLV
NSEGMENTS 2
SEGMENT   9lprA01:1:2
SRANGE    START=16  STOP=115
SLENGTH   74
SSEQH     >pdb|9lprA01:1:2
SSEQS     IVGGIEYSINNASLCSVGFSVTRGATKGFVTAGHCGTVNATARIGGAVVGTFAARVFPGNDRAWVSLTSA
SSEQS     QTLL
ENDSEG
SEGMENT   9lprA01:2:2
SRANGE    START=231  STOP=242
SLENGTH   13
SSEQH     >pdb|9lprA01:2:2
SSEQS     LQPILSQYGLSLV
ENDSEG
//

NOTE:
The following CATH hierarchy description lines are typically found together 
(as found in the CathDomainList and CathNames files)

CATHCODE  2.40.10.10
CLASS     Mainly Beta
ARCH      Beta Barrel
TOPOL     Thrombin, subunit H
HOMOL     Trypsin-like serine proteases

The following domain sequence lines are typically found together 
(as found in the CathDomain Fasta Sequence File)

DLENGTH   87
DSEQH     >pdb|9lprA01
DSEQS     IVGGIEYSINNASLCSVGFSVTRGATKGFVTAGHCGTVNATARIGGAVVGTFAARVFPGNDRAWVSLTSA
DSEQS     QTLLLQPILSQYGLSLV

Segment sequence lines are always initiated with a 'SEGMENT' tag
and terminated with an 'ENDSEG' tag. The number of segments in the domain
always precedes the first segment using the 'NSEGMENTS' tag.

The following segment sequence lines are typically found together
(as found in the CathSegments Fasta Sequence File)

SEGMENT   9lprA01:1:2
SRANGE    START=16  STOP=115
SLENGTH   74
SSEQH     >pdb|9lprA01:1:2
SSEQS     IVGGIEYSINNASLCSVGFSVTRGATKGFVTAGHCGTVNATARIGGAVVGTFAARVFPGNDRAWVSLTSA
SSEQS     QTLL
ENDSEG


CATH Domain and Segment Naming Conventions
==========================================
CATH Domain Names
-----------------
The domain names have seven characters (e.g. 1oaiA00).

CHARACTERS 1-4: PDB Code
The first 4 characters determine the PDB code e.g. 1oai

CHARACTER 5: Chain Character
This determines which PDB chain is represented.
Chain characters of zero ('0') indicate that the PDB file has no chain field.

CHARACTER 6-7: Domain Number
The domain number is a 2-figure, zero-padded number (e.g. '01', '02' ... '10', '11', '12'). Where the domain number is a double ZERO ('00') this indicates that the domain is a whole PDB chain with no domain chopping. 

CATH Segment Names
------------------
CATH segments (continuous regions of sequence within a domain) are described
adding colon separated numbers to the end of the domain name.
The first number is the sequential number of the segment.
The second number is the total number of segments in this domain.

1abcA01:1:2
xxxxxxxooooo

x = standard CATH six character domain name
o = segment information :ThisSegment:TotalSegments



New to CDDF Format 2.0
======================
Comments line is now optional

Domain names and segment names are based on the new CATH 7 character domain name format.

Where the HOMOL entry, naming the homologous superfamily level, is unnamed, the entry inherits the description from the parent topology level.
