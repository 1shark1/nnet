#! /usr/bin/perl	

# script converting rec.mapped to akulabs, input: wav list without extension

use File::Basename;
use File::Path qw(make_path);
	
print STDERR "Creating labels for file $ARGV[0]\n";

my $sameFolder = 1;

my $inputFolder = "/wav/";
my $akuFolder = "/akulab/";
my $mfccFolder="/fbc/";
my $labelFolder="/rec.mapped/";

my $osuff=".akulab";
my $datasuff=".par";
my $labsuff=".rec.mapped";

my @classes;
my $cnt=0;

open(HH,$ARGV[0]);
while(my $label=<HH>)
{
 chomp($label);
 $label.="$labsuff";
 if($sameFolder != 1) {
  $label=~s/$inputFolder/$labelFolder/i;
 }
 if($label=~/${labsuff}$/i)
  {
    #print "$label\n";
    $file=$label;	
    $file=~s/${labsuff}$/${datasuff}/i;
		if($sameFolder != 1) {
			$file=~s/$labelFolder/$mfccFolder/i;
		}

    $olabel=$label;
    $olabel=~s/${labsuff}$/${osuff}/i;
		if($sameFolder != 1) {
			$olabel=~s/$labelFolder/$akuFolder/i;
		}
    open(F,$file) or print STDERR "Can't open $file\n";
    binmode F;
    my $p;
    read(F,$p,12);
    my ($nSamples,$sampPeriod,$sampSize,$parmKind)=unpack("LLss", $p);
#    print "$file => $nSamples $sampSize\n";
    close(F);
    my @index=();
    open(F,$label) or print STDERR "Can't open $label\n";
	while(my $line=<F>)
	{
		#print "$line\n";
		if($line=~/^(\d+)\s+(\d+)\s+(\d+)/)
		{
#			print "AA: $1 $2 $3\n";
			for(my $i=$1;$i<$2;$i++)
			{
			   $index[$i]=$3;
			   $classes[$3]++;
			}
		}
	}	
    close(F);
#    print @index,"\n";
    if(scalar(@index) != $nSamples)
	{

		print STDERR printf("Unequal sizes %d != %d of index vs binary %s", scalar(@index),$nSamples,$label);
		next;
	}
		make_path(dirname($olabel));
    open(G,">$olabel") or die "Cant' print index $olabel";
    binmode G;
    print G pack ("(I)*",@index);
    close(G);

  }
  $cnt++;
# if($cnt==10)
 #  {last;}
}


#print "CLASS: COUNT\n";
open(HH, $ARGV[0] . "-framestats");
my $count = 0;
for(my $i=0;$i<@classes;$i++)
{
  print HH "$i: $classes[$i]\n";
  $count+=$classes[$i];
} 
print HH $count;
close HH;
 
