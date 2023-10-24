select case 
	when "Marital Status"='Married' then 'Menikah'
	when "Marital Status"='Single' then 'Belum Menikah'
	else 'Tidak diketahui'
end as "Status Pernikahan", avg(age) as "Rata Rata usia"
from customer
group by "Marital Status"