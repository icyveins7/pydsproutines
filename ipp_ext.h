#include "ipp.h"
#include <string>

#define NO_DEFAULT -1

namespace ippe
{
	const size_t INITIAL_CAP = 128;
	
	template <typename T>
	class vector
	{
		private:
			size_t numel;
			size_t cap;
			size_t copylen;
			bool reMalloc;
			T *m_data;
		
			void vector_base(size_t count)
			{
				// std::cout << "Base ctor for all specialized types." << std::endl;
				numel = count;
				cap = INITIAL_CAP;
				
				if (numel > cap){
					cap = numel;
				}
				reMalloc = false;
			}
		public:
			vector<T>(size_t count = 0);
			
			// common methods
			T* data(){
				return m_data;
			}
			
			T& back(){
				return m_data[numel-1];
			}
			
			T& front(){
				return m_data[0];
			}
			
			T& at(size_t pos)
			{
				if (pos < numel){
					return m_data[pos];
				}
				else{
					throw std::out_of_range(std::string("ippe::vector::range_check: Size is ") + std::to_string(numel));
				}
			}
			
			void push_back(T value) // for now lets not deal with lvalue/rvalue refs
			{
				// check size
				if (numel == cap){
					resize(cap * 2);
				}
				
				m_data[numel] = value;
				numel++;
			}
			
			size_t size()
			{
				return numel;
			}
			
			size_t capacity()
			{
				return cap;
			}
			
			void clear()
			{
				numel = 0;
			}
			
			bool empty()
			{
				if (numel == 0){return true;}
				else{ return false;}
			}
			
			void base_resize(size_t count)
			{
				// std::cout << "Base class resize." << std::endl;
				if (count > cap)
				{
					copylen = numel; // old, shorter length
					numel = count; // the new length
					cap = count; // capacity = new length
					reMalloc = true;
				}
				else{
					numel = count;
				}
			}
	
			void resize(size_t count);
			
			~vector()
			{
				// std::cout<<"Destructing base ipp vector."<<std::endl;
				
				ippsFree(m_data);
			}
	};
	
	// ========== specialized ctor
	
	// default ctor
	template <typename T>
	vector<T>::vector(size_t count)
	{
		// std::cout<<"There is no default template for IPP vectors. Please specify a valid IPP type." << std::endl;
		throw NO_DEFAULT;
	}
	
	// Ipp8u ctor
	template <>
	vector<Ipp8u>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp8u vector."<<std::endl;
		
		m_data = ippsMalloc_8u_L(cap);
	}
	
	// Ipp16u ctor
	template <>
	vector<Ipp16u>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp16u vector."<<std::endl;
		
		m_data = ippsMalloc_16u_L(cap);
	}
	
	// Ipp32u ctor
	template <>
	vector<Ipp32u>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp32u vector."<<std::endl;
		
		m_data = ippsMalloc_32u_L(cap);
	}
	
	// Ipp8s ctor
	template <>
	vector<Ipp8s>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp8s vector."<<std::endl;
		
		m_data = ippsMalloc_8s_L(cap);
	}
	
	// Ipp16s ctor
	template <>
	vector<Ipp16s>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp16s vector."<<std::endl;
		
		m_data = ippsMalloc_16s_L(cap);
	}
	
	// Ipp32s ctor
	template <>
	vector<Ipp32s>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp32s vector."<<std::endl;
		
		m_data = ippsMalloc_32s_L(cap);
	}
	
	// Ipp64s ctor
	template <>
	vector<Ipp64s>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp64s vector."<<std::endl;
		
		m_data = ippsMalloc_64s_L(cap);
	}
	
	// Ipp32f ctor
	template <>
	vector<Ipp32f>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp32f vector."<<std::endl;
		
		m_data = ippsMalloc_32f_L(cap);
	}
	
	// Ipp64f ctor
	template <>
	vector<Ipp64f>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp64f vector."<<std::endl;
		
		m_data = ippsMalloc_64f_L(cap);
	}
	
	// Ipp8sc ctor
	template <>
	vector<Ipp8sc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp8sc vector."<<std::endl;
		
		m_data = ippsMalloc_8sc_L(cap);
	}
	
	// Ipp16sc ctor
	template <>
	vector<Ipp16sc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp16sc vector."<<std::endl;
		
		m_data = ippsMalloc_16sc_L(cap);
	}
	
	// Ipp32sc ctor
	template <>
	vector<Ipp32sc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp32sc vector."<<std::endl;
		
		m_data = ippsMalloc_32sc_L(cap);
	}
	
	// Ipp64sc ctor
	template <>
	vector<Ipp64sc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp64sc vector."<<std::endl;
		
		m_data = ippsMalloc_64sc_L(cap);
	}
	
	// Ipp32fc ctor
	template <>
	vector<Ipp32fc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp32fc vector."<<std::endl;
		
		m_data = ippsMalloc_32fc_L(cap);
	}
	
	// Ipp64fc ctor
	template <>
	vector<Ipp64fc>::vector(size_t count)
	{
		vector_base(count);
		
		// std::cout<<"Constructing Ipp64fc vector."<<std::endl;
		
		m_data = ippsMalloc_64fc_L(cap);
	}
	
	// ========== specialized resize, but some of them don't exist
	
	// default resize
	template <typename T>
	void vector<T>::resize(size_t count)
	{
		// std::cout << "No default resize. This should never happen, should have been caught in ctor." << std::endl;
		throw NO_DEFAULT;
	}
	
	// Ipp8u resize
	template <>
	void vector<Ipp8u>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp8u specific resize. " << std::endl;
		if (reMalloc){
			Ipp8u *newm_data = ippsMalloc_8u_L(cap);
			ippsCopy_8u(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp16u resize, copy function does not exist
	template <>
	void vector<Ipp16u>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp16u specific resize. " << std::endl;
		if (reMalloc){
			Ipp16u *newm_data = ippsMalloc_16u_L(cap);
			ippsCopy_16s((Ipp16s*)m_data, (Ipp16s*)newm_data, copylen); // reuse 16s for non-existent 16u
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp32u resize, copy function does not exist
	template <>
	void vector<Ipp32u>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp32u specific resize. " << std::endl;
		if (reMalloc){
			Ipp32u *newm_data = ippsMalloc_32u_L(cap);
			ippsCopy_32s((Ipp32s*)m_data, (Ipp32s*)newm_data, copylen); // reuse 32s
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp8s resize, copy function does not exist
	template <>
	void vector<Ipp8s>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp8s specific resize. " << std::endl;
		if (reMalloc){
			Ipp8s *newm_data = ippsMalloc_8s_L(cap);
			ippsCopy_8u((Ipp8u*)m_data, (Ipp8u*)newm_data, copylen); // reuse 8u
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp16s resize
	template <>
	void vector<Ipp16s>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp16s specific resize. " << std::endl;
		if (reMalloc){
			Ipp16s *newm_data = ippsMalloc_16s_L(cap);
			ippsCopy_16s(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp32s resize
	template <>
	void vector<Ipp32s>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp32s specific resize. " << std::endl;
		if (reMalloc){
			Ipp32s *newm_data = ippsMalloc_32s_L(cap);
			ippsCopy_32s(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp64s resize
	template <>
	void vector<Ipp64s>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp64s specific resize. " << std::endl;
		if (reMalloc){
			Ipp64s *newm_data = ippsMalloc_64s_L(cap);
			ippsCopy_64s(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp32f resize
	template <>
	void vector<Ipp32f>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp32f specific resize. " << std::endl;
		if (reMalloc){
			Ipp32f *newm_data = ippsMalloc_32f_L(cap);
			ippsCopy_32f(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp64f resize
	template <>
	void vector<Ipp64f>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp64f specific resize. " << std::endl;
		if (reMalloc){
			Ipp64f *newm_data = ippsMalloc_64f_L(cap);
			ippsCopy_64f(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp8sc resize, copy function does not exist
	template <>
	void vector<Ipp8sc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp8sc specific resize. " << std::endl;
		if (reMalloc){
			Ipp8sc *newm_data = ippsMalloc_8sc_L(cap);
			ippsCopy_16s((Ipp16s*)m_data, (Ipp16s*)newm_data, copylen); // reuse 16s
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp16sc resize
	template <>
	void vector<Ipp16sc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp16sc specific resize. " << std::endl;
		if (reMalloc){
			Ipp16sc *newm_data = ippsMalloc_16sc_L(cap);
			ippsCopy_16sc(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp32sc resize
	template <>
	void vector<Ipp32sc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp32sc specific resize. " << std::endl;
		if (reMalloc){
			Ipp32sc *newm_data = ippsMalloc_32sc_L(cap);
			ippsCopy_32sc(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp64sc resize
	template <>
	void vector<Ipp64sc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp64sc specific resize. " << std::endl;
		if (reMalloc){
			Ipp64sc *newm_data = ippsMalloc_64sc_L(cap);
			ippsCopy_64sc(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp32fc resize
	template <>
	void vector<Ipp32fc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp32fc specific resize. " << std::endl;
		if (reMalloc){
			Ipp32fc *newm_data = ippsMalloc_32fc_L(cap);
			ippsCopy_32fc(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
	// Ipp64fc resize
	template <>
	void vector<Ipp64fc>::resize(size_t count)
	{
		base_resize(count);
				
		// std::cout << "Ipp64fc specific resize. " << std::endl;
		if (reMalloc){
			Ipp64fc *newm_data = ippsMalloc_64fc_L(cap);
			ippsCopy_64fc(m_data, newm_data, copylen);
			ippsFree(m_data);
			m_data = newm_data;
			
			reMalloc = false;
		}
	}
}